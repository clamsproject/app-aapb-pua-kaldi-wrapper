import argparse
import bisect
import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, Sequence, Tuple, Union

import ffmpeg
from clams import ClamsApp, Restifier
from lapps.discriminators import Uri
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

import metadata


class AAPB_PUA_Kaldi(ClamsApp):
    token_boundary = ' '
    silence_gap = 1.0  # seconds to insert between segments when patchworking
    timeunit_conv = {'milliseconds': 1000, 'seconds': 1}

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        if not isinstance(mmif, Mmif):
            mmif: Mmif = Mmif(mmif)

        # get AudioDocuments with locations
        docs = [document for document in mmif.documents if 
                document.at_type in (DocumentTypes.AudioDocument, DocumentTypes.VideoDocument) and 
                len(document.location) > 0]
        conf = self.get_configuration(**parameters)
        use_speech_segmentation = conf.get('use_speech_segmentation', True)

        if use_speech_segmentation:
            # using "speech" TimeFrames, `files` holds newly generated patchwork audio files in `tmpdir`
            files, tf_src_view, tmpdir = self._patchwork_audiofiles(mmif, docs)
        else:
            # `files` holds original locations
            files = {doc.id: doc.location_path() for doc in docs}
            tf_src_view = {}
            tmpdir = None

        transcript_tmpdir = self._run_kaldi(files)
        transcripts = transcript_tmpdir.name

        # now re-format Kaldi output
        self._kaldi_to_mmif(mmif, conf, transcripts, tf_src_view)
        
        if transcript_tmpdir:
            transcript_tmpdir.cleanup()
        if tmpdir:
            tmpdir.cleanup()
        return mmif

    def _kaldi_to_mmif(self, mmif, configuration, kaldi_out_dir, segmentataion_indices=None):
        for transcript_fname in os.listdir(kaldi_out_dir):  # files names after the ID of the AudioDocs
            with open(os.path.join(kaldi_out_dir, transcript_fname), encoding='utf8') as json_file:
                audiodoc_id = os.path.splitext(transcript_fname)[0]
                transcript = json.load(json_file)
                # convert transcript to MMIF view
                view: View = mmif.new_view()
                self.sign_view(view, configuration)
                view.new_contain(DocumentTypes.TextDocument)
                view.new_contain(Uri.TOKEN)
                view.new_contain(AnnotationTypes.TimeFrame, 
                                 timeUnit=metadata.timeunit,
                                 document=audiodoc_id)
                view.new_contain(AnnotationTypes.Alignment)
                if segmentataion_indices is None or len(segmentataion_indices) == 0:
                    self._kaldi_to_single_textdocument(transcript, view, mmif.get_document_by_id(audiodoc_id))
                else:
                    self._kaldi_to_segmented_textdocument(transcript, view, segmentataion_indices[audiodoc_id])
    
    def _kaldi_to_single_textdocument(self, pua_transcript, view, source_audio_doc):
        """
        Given a PUA transcript, create a single TextDocument and align it to the source audio.
        
        :param pua_transcript: PUA transcript in python dict
        :param view: MMIF view to add annotations to
        :param source_audio_doc: source AudioDocument
        
        """
        # PUA transcript has this structure;
        """
        {
            "words": [
                { "time": 14.45, "word": "no", "duration": "0.15" },
                { "duration": "0.27", "word": "no", "time": 18.95 }, 
                ...
            ]
        }
        """
        # join tokens
        raw_text = self.token_boundary.join([token['word'] for token in pua_transcript['words']])
        # make annotations
        textdoc = view.new_textdocument(raw_text)
        view.new_annotation(AnnotationTypes.Alignment, source=source_audio_doc.id, target=textdoc.id)
        char_offset = 0
        for index, word_obj in enumerate(pua_transcript['words']):
            raw_token = word_obj['word']
            tok_start = char_offset
            tok_end = tok_start + len(raw_token)
            char_offset += len(raw_token) + len(self.token_boundary)
            token = view.new_annotation(Uri.TOKEN, 
                                        start=tok_start, end=tok_end, 
                                        word=word_obj['word'], 
                                        document=f'{view.id}:{textdoc.id}')
            tf_start = int(word_obj['time'] * self.timeunit_conv[metadata.timeunit])
            tf_end = int(float(word_obj['duration']) * self.timeunit_conv[metadata.timeunit]) + tf_start
            tf = view.new_annotation(AnnotationTypes.TimeFrame, start=tf_start, end=tf_end, frameType='speech')
            view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)  # counting one for TextDoc-AudioDoc alignment

    def _kaldi_to_segmented_textdocument(self, transcript, view: View, view_w_tf: View):
        segment_ids, original_starts, original_ends, patchwork_starts, patchwork_ends = \
            self._align_segmentations_to_patchwork(view_w_tf.get_annotations(AnnotationTypes.TimeFrame, frameType='speech'))
        
        cur_segment = 0
        # start with an empty text doc to provide text doc id to token annotations
        raw_text = ""
        textdoc = view.new_textdocument(raw_text)
        position = 0
        sorted_words = sorted(transcript['words'], key=lambda x:x['time'])
        for index, word_obj in enumerate(sorted_words):
            # if index == len(transcript['words']):
            raw_token = word_obj['word']
            # this time point is bound to the "patchwork" audio stream
            start_in_patchwork = word_obj['time'] * self.timeunit_conv[metadata.timeunit]
            end_in_patchwork = float(word_obj['duration']) * self.timeunit_conv[metadata.timeunit] + start_in_patchwork

            # and count characters
            char_start = position
            char_end = char_start + len(raw_token)
            position += len(raw_token) + len(self.token_boundary)

            # figure out in which n-th segment this token was
            segment_num = bisect.bisect(patchwork_starts, start_in_patchwork) - 1
            # next, check the token actually fall into the segment
            if not (patchwork_ends[segment_num] > start_in_patchwork and
                    patchwork_ends[segment_num] > end_in_patchwork):
                # if not, just ignore this token, as it's probably kaldi's fault
                # (e.g. recognizing something from a silence gap)
                continue

            # when moved on to the next speech segment
            if segment_num > cur_segment:
                # inject collected tokens into the text doc and finalize it
                textdoc.text_value = raw_text
                view.new_annotation(AnnotationTypes.Alignment, source=view_w_tf.annotations.get(segment_ids[cur_segment]).id, target=textdoc.id)

                # reset stuff and start a new text doc
                position = 0
                cur_segment = segment_num
                raw_text = ""
                textdoc = view.new_textdocument(raw_text)

            # regardless of speech segment, process individual tokens
            token = view.new_annotation(Uri.TOKEN,
                                        start=char_start, end=char_end,
                                        word=raw_token,
                                        document=f'{view.id}:{textdoc.id}')
            offset_from_original = original_starts[segment_num] - patchwork_starts[segment_num]
            start = int(start_in_patchwork + offset_from_original) 
            end = int(end_in_patchwork + offset_from_original)
            # TODO (krim @ 11/30/20): what happens when kaldi recognized a token spreads over to a "silence" zone?
            tf = view.new_annotation(AnnotationTypes.TimeFrame, start=start, end=end, frameType='speech')
            view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)
            if len(raw_text) == 0:
                raw_text = raw_token
            else:
                raw_text = self.token_boundary.join((raw_text, raw_token))
        textdoc.text_value = raw_text
        view.new_annotation(AnnotationTypes.Alignment, source=view_w_tf.annotations.get(segment_ids[cur_segment]).id, target=textdoc.id)

    def _align_segmentations_to_patchwork(self, speech_segment_annotations):
        speech_segments = [(ann.id, ann.properties['start'], ann.properties['end'])
                           for ann in speech_segment_annotations]

        sorted_segments = sorted(speech_segments, key=lambda x: x[1])
        segment_ids, ori_starts, ori_ends = zip(*sorted_segments)
        # new time offsets in the "patchwork" audio
        new_starts = []
        new_ends = []
        for i in range(len(ori_starts)):
            if i == 0:
                new_starts.append(0)
            else:
                new_starts.append(new_ends[i-1] + self.silence_gap * self.timeunit_conv[metadata.timeunit])
            new_ends.append(ori_ends[i] - ori_starts[i] + new_starts[i])
        return segment_ids, ori_starts, ori_ends, new_starts, new_ends

    def _patchwork_audiofiles(self, mmif, audio_documents):
        """ 
        Creates patchwork audio files from full audio files and "speech" TimeFrames.
        """
        files = {}
        tf_src_view = {}
        # this tmp dir must be created at every `annotate` call
        patchwork_dir = tempfile.TemporaryDirectory()
        for audio_document in audio_documents:
            segment_views = [view for view in mmif.get_views_for_document(audio_document.id)
                             if AnnotationTypes.TimeFrame in view.metadata.contains]
            if len(segment_views) > 1:
                # TODO (krim @ 11/30/20): we might want to actually handle 
                # this situation; e.g. for evaluating multiple segmenter
                raise ValueError('got multiple segmentation views for a document with TimeFrames')
            elif len(segment_views) == 1:
                view = segment_views[0]
                timeunit = view.metadata.contains[AnnotationTypes.TimeFrame]['timeUnit']
                # start & end in this list should be converted into seconds 
                # for ffmpeg to work
                segments = [(int(ann.properties['start']) / self.timeunit_conv[timeunit],
                             int(ann.properties['end']) / self.timeunit_conv[timeunit])
                            for ann in view.get_annotations(AnnotationTypes.TimeFrame, frameType='speech')]
                patchwork_fname = os.path.join(patchwork_dir.name, audio_document.id + '.wav')
                self._patchwork_audiofile(audio_document.location_path(), patchwork_fname, segments)
                files[audio_document.id] = patchwork_fname
                tf_src_view[audio_document.id] = view
            elif len(segment_views) == 0:
                files[audio_document.id] = audio_document.location_path()
        return files, tf_src_view, patchwork_dir

    def _patchwork_audiofile(self, in_fname: str, out_fname: str,
                             indexed_speech_segments: Sequence[Tuple[float, float]], dryrun=False):
        """
        Given a "full" audio file and list of speech segmentations, this will 
        create a "patchwork" audio file that has only speech parts slices and 
        put together, using ffmpeg "atrim" filter. Note that between speech 
        parts, a short gap of silence will be inserted (set by self.silence_gap).
        
        :param in_fname: file name of input 
        :param out_fname: file name of output
        :param indexed_speech_segments: Start and end time points of speech segments.
                                        Start and end must be in seconds (in decimal),
                                        as ffmpeg uses seconds.
        :param dryrun: When true, just print out ffmpeg command, not actually running it
        """
        original_audio = ffmpeg.input(in_fname)
        silence = ffmpeg.input('anullsrc', f='lavfi')
        patches = []
        silences = silence.filter_multi_output('asplit')
        for i, (start, end) in enumerate(indexed_speech_segments):
            silence = silences[i]
            gap = silence.filter('atrim', duration=self.silence_gap)
            patches.append(original_audio.filter('atrim', start=start, end=end))
            patches.append(gap)
        ffmpeg_cmd = ffmpeg.concat(*patches, v=0, a=1)
        ffmpeg_cmd = ffmpeg_cmd.output(out_fname)
        # for debugging
        if dryrun:
            print(' '.join(ffmpeg_cmd.compile()))
        else:
            ffmpeg_cmd.run(overwrite_output=True)
            
    @staticmethod
    def _run_kaldi(files: Dict[str, str]) -> tempfile.TemporaryDirectory:
        """
        Run AAPB-PUA kaldi as a subprocess on input files.
        
        :param files: dict of {AudioDocument.id : physical file location}
        :return: A TemporaryDirectory where automatic transcripts are stored.
                 Each transcript is named after the source audio document id.
        """
        # files has full path to files as keys and ID of the corresponding AudioDoc as values

        # make a temporary dir for kaldi-ready audio files
        audio_tmpdir = tempfile.TemporaryDirectory()
        # make another temporary dir to store resulting .json files
        trans_tmpdir = tempfile.TemporaryDirectory()

        def puakaldi_exp_dir(kaldi_root):
            return os.path.join(kaldi_root, 'egs', 'american-archive-kaldi', 'sample_experiment')
        
        # Steve's kaldi wrapper (run_kaldi.py) does: 
        # 1. cd to kaldi_exp_dir
        # 2. validate necessary files 
        # 3. create `output` in the kaldi_exp_dir
        # 4. for each wav_file, $(kaldi_exp_dir/run.sh $wav_file $out_json_file)
        # 5. convert json into plain txt transcript
        # Because step 1, 2, 3, 5 are not necessary, we are bypassing `run_kaldi.py` 
        # and directly call the main kaldi pipeline (run.sh)

        for audio_docid, audio_fname in files.items():
            resampled_audio_fname = f'{audio_tmpdir.name}/{audio_docid}_16kHz.wav'
            result_transcript_fname = f'{trans_tmpdir.name}/{audio_docid}.json'
            # resample to a single-channel, 16k wav file
            ffmpeg.input(audio_fname).output(resampled_audio_fname, ac=1, ar=16000).run()
            subprocess.run([
                f'{puakaldi_exp_dir(os.getenv("KALDI_ROOT")) if "KALDI_ROOT" in os.environ else "/opt/kaldi"}/run.sh',
                resampled_audio_fname,
                result_transcript_fname
            ], check=True)
        audio_tmpdir.cleanup()
        return trans_tmpdir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', action='store', default='5000', help='set port to listen')
    parser.add_argument('--production', action='store_true', help='run gunicorn server')
    parsed_args = parser.parse_args()

    puakaldi = AAPB_PUA_Kaldi()
    puakaldi_flask = Restifier(puakaldi, port=int(parsed_args.port))
    if parsed_args.production:
        puakaldi_flask.serve_production()
    else:
        puakaldi.logger.setLevel(logging.DEBUG)
        puakaldi_flask.run()
