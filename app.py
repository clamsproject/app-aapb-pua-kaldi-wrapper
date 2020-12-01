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

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes, Text
from lapps.discriminators import Uri

APP_VERSION = '0.1.0'
WRAPPED_IMAGE = 'brandeisllc/aapb-pua-kaldi:v1'
TOKEN_PREFIX = 't'
TEXT_DOCUMENT_PREFIX = 'td'
TIME_FRAME_PREFIX = 'tf'
ALIGNMENT_PREFIX = 'a'
TRANSCRIPT_DIR = "output"

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

def kaldi_exp_dir(kaldi_root):
    return os.path.join(kaldi_root, 'egs', 'american-archive-kaldi', 'sample_experiment')


class Kaldi(ClamsApp):

    def setupmetadata(self) -> dict:
        return {
            "name": "Kaldi Wrapper",
            "description": "This tool wraps the Kaldi ASR tool",
            "vendor": "Team CLAMS",
            "iri": f"http://mmif.clams.ai/apps/kaldi/{APP_VERSION}",
            "wrappee": WRAPPED_IMAGE,
            "requires": [DocumentTypes.AudioDocument.value],
            "produces": [
                DocumentTypes.TextDocument.value,
                AnnotationTypes.TimeFrame.value,
                AnnotationTypes.Alignment.value,
                Uri.TOKEN
            ]
        }

    def sniff(self, mmif) -> bool:
        if type(mmif) is not Mmif:
            mmif = Mmif(mmif)
        return len(mmif.get_documents_locations(DocumentTypes.AudioDocument.value)) > 0

    def annotate(self, mmif: Union[str, dict, Mmif], run_kaldi=True, pretty=False) -> str:
        mmif_obj: Mmif
        if isinstance(mmif, Mmif):
            mmif_obj: Mmif = mmif
        else:
            mmif_obj: Mmif = Mmif(mmif)

        # get AudioDocuments with locations
        docs = [document for document in mmif_obj.documents
                if document.at_type == DocumentTypes.AudioDocument.value and len(document.location) > 0]

        files = {doc.id: doc.location for doc in docs}

        # key them by location basenames
        docs_dict: Dict[str, Document] = {os.path.splitext(os.path.basename(doc.location))[0]: doc for doc in docs}
        assert len(docs) == len(docs_dict), 'no duplicate filenames'
        # TODO (angus-lherrou @ 2020-10-03): allow duplicate basenames for files originally from different folders
        #  by renaming files more descriptively


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
                # join tokens
                whitespace = ' '
                raw_text = whitespace.join([token['word'] for token in transcript['words']])
                # make annotations
                textdoc = self.create_td(raw_text, 0)
                view.add_document(textdoc)
                align_1 = self.create_align(mmif_obj.get_document_by_id(audiodoc_id), textdoc, 0)
                view.add_annotation(align_1)
                position = 0
                for index, word_obj in enumerate(transcript['words']):
                    raw_token = word_obj['word']
                    start = position
                    end = start + len(raw_token)
                    position += len(raw_token) + len(whitespace)
                    tf = self.create_tf(word_obj['time'], word_obj['duration'], index)
                    raw_token = self.create_token(word_obj['word'], index, start, end, f'{view.id}:{textdoc.id}')
                    align = self.create_align(tf, raw_token, index+1)  # counting one for TextDoc-AudioDoc alignment
                    view.add_annotation(raw_token)
                    view.add_annotation(tf)
                    view.add_annotation(align)

        if transcript_tmpdir:
            transcript_tmpdir.cleanup()
        return mmif_obj.serialize(pretty=pretty)

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
        token.add_property('start', str(start))
        token.add_property('end', str(end))
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
                        help='Use this flag if you want to run Kaldi on a path you specify, instead of running '
                             'the Flask app.')
    parser.add_argument('--no-kaldi',
                        action='store_false',
                        help='Add this flag if Kaldi has already been run and you just want to re-annotate.')
    parser.add_argument('--pretty',
                        action='store_true',
                        help='Use this flag to return "pretty" (indented) MMIF data.')

    parsed_args = parser.parse_args()

    if parsed_args.once:
        with open(parsed_args.once) as mmif_in:
            mmif_str = mmif_in.read()

        kaldi_app = Kaldi()

        mmif_out = kaldi_app.annotate(mmif_str, run_kaldi=parsed_args.no_kaldi, pretty=parsed_args.pretty)
        with open('mmif_out.json', 'w') as out_file:
            out_file.write(mmif_out)
    else:
        kaldi_app = Kaldi()
        annotate = kaldi_app.annotate
        kaldi_app.annotate = lambda *args, **kwargs: annotate(*args,
                                                              run_kaldi=parsed_args.no_kaldi,
                                                              pretty=parsed_args.pretty)
        kaldi_service = Restifier(kaldi_app)
        kaldi_service.run()


if __name__ == '__main__':
    main()
