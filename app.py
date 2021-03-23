# app-puakaldi-wrapper version 0.1.0
# author: Angus L'Herrou
# org: CLAMS team

import ffmpeg
import json
import os
import subprocess
from typing import Dict, Sequence, Tuple, List, Union
import argparse
import tempfile
import bisect

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes, Text
from lapps.discriminators import Uri

__version__ = '0.2.0'
WRAPPED_IMAGE = 'brandeisllc/aapb-pua-kaldi:v1'
TOKEN_PREFIX = 't'
TEXT_DOCUMENT_PREFIX = 'td'
TIME_FRAME_PREFIX = 'tf'
ALIGNMENT_PREFIX = 'a'
TRANSCRIPT_DIR = "output"
SILENCE_GAP_LEN = 1 # seconds to insert between segments when trimming


__ALL__ = [
    "WRAPPED_IMAGE",
    "TOKEN_PREFIX",
    "TEXT_DOCUMENT_PREFIX",
    "TIME_FRAME_PREFIX",
    "ALIGNMENT_PREFIX",
    "TRANSCRIPT_DIR",
    "Kaldi",
    "kaldi"
]

TIME_UNITS = {
    'milliseconds': 1000,
    'seconds': 1
}


def kaldi_exp_dir(kaldi_root):
    return os.path.join(kaldi_root, 'egs', 'american-archive-kaldi', 'sample_experiment')


class Kaldi(ClamsApp):

    def _appmetadata(self) -> dict:
        return {
            "name": "Kaldi Wrapper",
            "description": "This tool wraps the Kaldi ASR tool",
            "vendor": "Team CLAMS",
            "iri": f"http://mmif.clams.ai/apps/kaldi/{__version__}",
            "wrappee": WRAPPED_IMAGE,
            "requires": [DocumentTypes.AudioDocument.value],
            "produces": [
                DocumentTypes.TextDocument.value,
                AnnotationTypes.TimeFrame.value,
                AnnotationTypes.Alignment.value,
                Uri.TOKEN
            ]
        }

    def _annotate(self, mmif: Union[str, dict, Mmif], run_kaldi=True, use_segmentation=True) -> Mmif:
        mmif_obj: Mmif
        if isinstance(mmif, Mmif):
            mmif_obj: Mmif = mmif
        else:
            mmif_obj: Mmif = Mmif(mmif)

        # get AudioDocuments with locations
        docs = [document for document in mmif_obj.documents
                if document.at_type == DocumentTypes.AudioDocument.value and len(document.location) > 0]
        files = {}
        tf_src_view = {}
        trimming_tmpdir = tempfile.TemporaryDirectory()

        for doc in docs:
            if use_segmentation:
                segment_views = [view for view in mmif_obj.get_views_for_document(doc.id) if AnnotationTypes.TimeFrame.value in view.metadata.contains]
                if len(segment_views) > 1:
                    # TODO (krim @ 11/30/20): we might want to actually handle this situation; e.g. for evaluating multiple segmenter
                    raise ValueError('got multiple segmentation views for a document with TimeFrames')
                elif len(segment_views) == 1:
                    view = segment_views[0]
                    segments = [(int(ann.properties['start']), int(ann.properties['end'])) for ann in view.get_annotations(AnnotationTypes.TimeFrame, frameType='speech')]
                    unit = view.metadata.contains[AnnotationTypes.TimeFrame.value]['unit']
                    trimmed_fname = os.path.join(trimming_tmpdir.name, doc.id + '.wav')
                    slice_and_merge_audio(trimmed_fname, doc.location_path(), segments, unit=unit, silence_gap=SILENCE_GAP_LEN)
                    files[doc.id] = trimmed_fname
                    tf_src_view[doc.id] = view.id
                elif len(segment_views) == 0:
                    files[doc.id] = doc.location_path()
            else:
                files[doc.id] = doc.location_path()

        transcript_tmpdir = None
        if run_kaldi:
            transcript_tmpdir = kaldi(files)
            transcripts = transcript_tmpdir.name
        else:
            transcripts = TRANSCRIPT_DIR

        # get Kaldi's output
        for transcript_fname in os.listdir(transcripts): # files names after the ID of the AudioDocs
            with open(os.path.join(transcripts, transcript_fname), encoding='utf8') as json_file:
                audiodoc_id = os.path.splitext(transcript_fname)[0]
                transcript = json.load(json_file)
                # convert transcript to MMIF view
                view: View = mmif_obj.new_view()
                self.stamp_view(view, audiodoc_id)
                whitespace = ' '
                position = 0
                if audiodoc_id in tf_src_view:
                    td_idx = 0
                    alignment_idx = 0
                    # the view with time frame
                    view_wtf = mmif_obj.get_view_by_id(tf_src_view[audiodoc_id])
                    segments = {ann.id: (int(ann.properties['start']), int(ann.properties['end'])) for ann in view_wtf.get_annotations(AnnotationTypes.TimeFrame, frameType='speech')}

                    # what the below is doing is basically create 'zippable' lists
                    # (segment_ids, ostarts, oends, nstarts, nends)
                    # that are all aligned to the order of speech segments found in the view_wtf
                    segment_ids = sorted(segments.keys()) # assuming segments are sorted by the start points
                    # original start points and end points
                    ostarts = sorted([s for s, _ in segments.values()])
                    oends = sorted([e for _, e in segments.values()])
                    # new time offsets in the "trimmed" audio
                    nstarts = []
                    nends = []
                    for i in range(len(ostarts)):
                        if i == 0:
                            nstarts.append(0)
                        else:
                            nstarts.append(nends[i-1]+SILENCE_GAP_LEN)
                        nends.append(oends[i] - ostarts[i] + nstarts[i])

                    # and put the cursor to the first speech segment
                    cur_segment = 0
                    # start with an empty text doc to provide text doc id to token annotations
                    raw_text = ""
                    textdoc = self.create_td(raw_text, td_idx)
                    td_idx += 1
                    for index, word_obj in enumerate(transcript['words']):
                        raw_token = word_obj['word']
                        # this time point is bound to the "trimmed" audio stream
                        time_start = word_obj['time']
                        # figure out in which n-th segment this token was
                        segment_num = bisect.bisect(nstarts, time_start) - 1

                        # and count characters
                        char_start = position
                        char_end = char_start + len(raw_token)
                        position += len(raw_token) + len(whitespace)
                        # next, figure out this token is kaldi's recognition of silence gap we used to buffer two segments in "trimmed" audio
                        # if so, ignore this token
                        if nends[segment_num] < time_start:
                            continue
                        # when moved on to the next speech segment
                        if segment_num - 1 == cur_segment:
                            # inject collected tokens into the text doc and finalize it
                            del raw_text[-1]
                            textdoc.properties.text_language = 'en'
                            textdoc.properties.text_value = raw_text
                            view.add_document(textdoc)
                            td_tf_alignment = self.create_align(segment_ids[cur_segment], textdoc, alignment_idx)
                            alignment_idx += 1
                            view.add_annotation(td_tf_alignment)

                            # reset stuff and start a new text doc
                            position = 0
                            cur_segment += 1
                            raw_text = ""
                            textdoc = self.create_td(raw_text, td_idx)
                            td_idx += 1

                        # regardless of speech segment, process individual tokens
                        token = self.create_token(raw_token, index, char_start, char_end, f'{view.id}:{textdoc.id}')
                        # convert the time stamp to the original audio
                        original_time_start = time_start + ostarts[segment_num] - nstarts[segment_num]
                        # TODO (krim @ 11/30/20): what happens when a kaldi recognized token spreads over to a "silence" zone?
                        tf = self.create_tf(original_time_start, word_obj['duration'], index)
                        tk_tf_alignment = self.create_align(tf, token, alignment_idx)
                        alignment_idx += 1
                        view.add_annotation(token)
                        view.add_annotation(tf)
                        view.add_annotation(tk_tf_alignment)
                        raw_text += raw_token + whitespace
                else:
                    # join tokens
                    raw_text = whitespace.join([token['word'] for token in transcript['words']])
                    # make annotations
                    textdoc = self.create_td(raw_text, 0)
                    view.add_document(textdoc)
                    align_1 = self.create_align(mmif_obj.get_document_by_id(audiodoc_id), textdoc, 0)
                    view.add_annotation(align_1)
                    for index, word_obj in enumerate(transcript['words']):
                        raw_token = word_obj['word']
                        start = position
                        end = start + len(raw_token)
                        position += len(raw_token) + len(whitespace)
                        tf = self.create_tf(word_obj['time'], word_obj['duration'], index)
                        token = self.create_token(word_obj['word'], index, start, end, f'{view.id}:{textdoc.id}')
                        align = self.create_align(tf, token, index+1)  # counting one for TextDoc-AudioDoc alignment
                        view.add_annotation(token)
                        view.add_annotation(tf)
                        view.add_annotation(align)

        if transcript_tmpdir:
            transcript_tmpdir.cleanup()
        trimming_tmpdir.cleanup()
        return mmif_obj

    @staticmethod
    def create_td(doc: str, index: int) -> Document:
        text = Text()
        text.value = doc
        td = Document()
        td.id = TEXT_DOCUMENT_PREFIX + str(index + 1)
        td.at_type = DocumentTypes.TextDocument.value
        td.properties.text = text
        return td

    @staticmethod
    def create_token(word: str, index: int, start: int, end: int, source_doc_id: str) -> Annotation:
        token = Annotation()
        token.at_type = Uri.TOKEN
        token.id = TOKEN_PREFIX + str(index + 1)
        token.add_property('word', word)
        token.add_property('start', start)
        token.add_property('end', end)
        token.add_property('document', source_doc_id)
        return token

    @staticmethod
    def create_tf(time: float, duration: str, index: int) -> Annotation:
        tf = Annotation()
        tf.at_type = AnnotationTypes.TimeFrame.value
        tf.id = TIME_FRAME_PREFIX + str(index + 1)
        tf.properties['frameType'] = 'speech'
        # times should be in milliseconds
        tf.properties['start'] = int(time * 1000)
        tf.properties['end'] = int((time + float(duration)) * 1000)
        return tf

    @staticmethod
    def create_align(source: Annotation, target: Annotation, index: int) -> Annotation:
        align = Annotation()
        align.at_type = AnnotationTypes.Alignment.value
        align.id = ALIGNMENT_PREFIX + str(index + 1)
        align.properties['source'] = source.id
        align.properties['target'] = target.id
        return align

    def stamp_view(self, view: View, tf_source_id: str) -> None:
        if view.is_frozen():
            raise ValueError("can't modify an old view")
        view.metadata['app'] = self.metadata['iri']
        view.new_contain(DocumentTypes.TextDocument.value)
        view.new_contain(Uri.TOKEN)
        view.new_contain(AnnotationTypes.TimeFrame.value, {'unit': 'milliseconds', 'document': tf_source_id})
        view.new_contain(AnnotationTypes.Alignment.value)


def slice_and_merge_audio(out_fname: str,
                          in_fname: str,
                          indexed_speech_segments: Sequence[Tuple[int, int]],
                          unit='milliseconds',
                          silence_gap=1,
                          dryrun=False):
    # this gap will be inserted after each segment when merging back to a single audio file
    original_audio = ffmpeg.input(in_fname)
    silence = ffmpeg.input('anullsrc', f='lavfi')
    to_concat = []
    silences = silence.filter_multi_output('asplit')
    for i, (start, end) in enumerate(indexed_speech_segments):
        start = start / TIME_UNITS[unit]
        end = end / TIME_UNITS[unit]
        silence = silences[i]
        gap = silence.filter('atrim', duration=silence_gap)
        to_concat.append(original_audio.filter('atrim', start=start, end=end))
        to_concat.append(gap)
    print(to_concat)
    del to_concat[-1] # fence posting
    cmd = ffmpeg.concat(*to_concat, v=0, a=1)
    cmd = cmd.output(out_fname)
    # for debugging
    if dryrun:
        print(' '.join(cmd.compile()))
    else:
        cmd.run(overwrite_output=True)


def kaldi(files: Dict[str, str]) -> tempfile.TemporaryDirectory:
    # files has full path to files as keys and ID of the corresponding AudioDoc as values

    # make a temporary dir for kaldi-ready audio files
    audio_tmpdir = tempfile.TemporaryDirectory()
    # make another temporary dir to store resulting .json files
    trans_tmpdir = tempfile.TemporaryDirectory()

    # Steve's kaldi wrapper (run_kaldi.py) does: 
    # 1. cd to kaldi_exp_dir
    # 2. validate necessary files 
    # 3. create `output` in the kaldi_exp_dir
    # 4. for each wav_file, $(kaldi_exp_dir/run.sh $wav_file $out_json_file)
    # 5. convert json into plain txt transcript
    # Because step 1, 2, 3, 5 are not necessary, we are bypassing `run_kaldi.py` and directly call the main kaldi pipeline (run.sh)

    for audio_docid, audio_fname in files.items():
        resampled_audio_fname = f'{audio_tmpdir.name}/{audio_docid}_16kHz.wav'
        result_transcript_fname = f'{trans_tmpdir.name}/{audio_docid}.json'
        # resample to a single-channel, 16k wav file
        ffmpeg.input(audio_fname).output(resampled_audio_fname, ac=1, ar=16000).run()
        subprocess.run([
            f'{kaldi_exp_dir(os.getenv("KALDI_ROOT")) if "KALDI_ROOT" in os.environ else "/opt/kaldi"}/run.sh', 
            resampled_audio_fname,
            result_transcript_fname
            ])
    audio_tmpdir.cleanup()
    return trans_tmpdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once',
                        type=str,
                        metavar='PATH',
                        help='Flag to run Kaldi on a specified MMIF, instead of running the Flask app.')
    parser.add_argument('--no-kaldi',
                        action='store_false',
                        help='Add this flag if Kaldi has already been run and you just want to re-annotate.')

    parsed_args = parser.parse_args()

    if parsed_args.once:
        with open(parsed_args.once) as mmif_in:
            mmif_str = mmif_in.read()

        kaldi_app = Kaldi()

        mmif_out = kaldi_app.annotate(mmif_str, run_kaldi=parsed_args.no_kaldi)
        with open('mmif_out.json', 'w') as out_file:
            out_file.write(mmif_out)
    else:
        kaldi_app = Kaldi()
        annotate = kaldi_app.annotate
        kaldi_app.annotate = lambda *args, **kwargs: annotate(*args,
                                                              run_kaldi=parsed_args.no_kaldi)
        kaldi_service = Restifier(kaldi_app)
        kaldi_service.run()


if __name__ == '__main__':
    main()
