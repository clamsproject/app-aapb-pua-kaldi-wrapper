# app-puakaldi-wrapper version 0.1.0
# author: Angus L'Herrou
# org: CLAMS team

import json
import os
import shutil
import subprocess
from typing import Dict, Sequence, Tuple, List, Union
import argparse
import tempfile

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes, Text
from lapps.discriminators import Uri

KALDI_AMERICAN_ARCHIVE = '/kaldi/egs/american-archive-kaldi'
KALDI_EXPERIMENT_DIR = os.path.join(KALDI_AMERICAN_ARCHIVE, 'sample_experiment')
APP_VERSION = '0.1.0'
WRAPPED_IMAGE = 'hipstas/kaldi-pop-up-archive:v1'
TOKEN_PREFIX = 't'
TEXT_DOCUMENT_PREFIX = 'td'
TIME_FRAME_PREFIX = 'tf'
ALIGNMENT_PREFIX = 'a'


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

        files = [document.location for document in docs]

        # key them by location basenames
        docs_dict: Dict[str, Document] = {os.path.splitext(os.path.basename(doc.location))[0]: doc for doc in docs}
        assert len(docs) == len(docs_dict), 'no duplicate filenames'
        # TODO (angus-lherrou @ 2020-10-03): allow duplicate basenames for files originally from different folders
        #  by renaming files more descriptively

        if run_kaldi:
            transcript_tmpdir = kaldi(files)

        # get Kaldi's output
        json_transcripts: Dict[str, dict] = {}
        for transcript in os.listdir(transcript_tmpdir.name):
            with open(os.path.join(transcript_tmpdir.name, transcript), encoding='utf8') as json_file:
                filename = os.path.splitext(transcript)[0]
                if filename.endswith('_16kHz'):
                    filename = filename[:-6]
                json_transcripts[filename] = json.load(json_file)

        assert sorted(docs_dict.keys()) == sorted(json_transcripts.keys()), 'got a transcript for every file'

        for basename, transcript in json_transcripts.items():
            # convert transcript to MMIF view
            view: View = mmif_obj.new_view()
            self.stamp_view(view, docs_dict[basename].id)
            # index and join tokens
            indices, doc = self.index_and_join_tokens([token['word'] for token in transcript['words']])
            # make annotations
            td = self.create_td(doc, 0)
            view.add_document(td)
            align_1 = self.create_align(docs_dict[basename], td, 0)
            view.add_annotation(align_1)
            for index, word_obj in enumerate(transcript['words']):
                tf = self.create_tf(word_obj['time'], word_obj['duration'], index)
                token = self.create_token(word_obj['word'], index, indices, f'{view.id}:{td.id}')
                align = self.create_align(tf, token, index+1)  # one more alignment than the others
                view.add_annotation(token)
                view.add_annotation(tf)
                view.add_annotation(align)

        transcript_tmpdir.cleanup()
        return mmif_obj.serialize(pretty=pretty)

    @staticmethod
    def index_and_join_tokens(tokens: Sequence[str]) -> Tuple[List[Tuple[int, int]], str]:
        position = 0
        indices: List[Tuple[int, int]] = []
        for token in tokens:
            start = position
            position += len(token)
            end = position
            position += 1
            indices.append((start, end))
        doc = ' '.join(tokens)
        return indices, doc

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
    def create_token(word: str, index: int, indices: List[Tuple[int, int]], source_doc_id: str) -> Annotation:
        token = Annotation()
        token.at_type = Uri.TOKEN
        token.id = TOKEN_PREFIX + str(index + 1)
        token.add_property('word', word)
        token.add_property('start', indices[index][0])
        token.add_property('end', indices[index][1])
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


def kaldi(files: list) -> None:

    # make a temporary dir for kaldi-ready audio files
    audio_tmpdir = tempfile.TemporaryDirectory()
    # make another temporary dir to store resulting .json files
    trans_tmpdir = tempfile.TemporaryDirectory()

    # Steve's kaldi wrapper (run_kaldi.py) does: 
    # 1. cd to KALDI_EXPERIMENT_DIR
    # 2. validate necessary files 
    # 3. create `output` in the KALDI_EXPERIMENT_DIR
    # 4. for each wav_file, $(KALDI_EXPERIMENT_DIR/run.sh $wav_file $out_json_file)
    # 5. convert json into plain txt transcript
    # Because step 1, 2, 4, 5 are not necessary, we are bypassing `run_kaldi.py` and directly call the main kaldi pipeline (run.sh)

    for audio_name in files: 
        audio_basename = os.path.splitext(os.paht.basename(audio_name))[0]
        subprocess.run(['ffmpeg', '-i', link, '-ac', '1', '-ar', '16000',
                         f'{audio_tmpdir.name}/{audio_basename}_16kHz.wav'])
        subprocess.run([
            f'{KALDI_EXPERIMENT_DIR}/run.sh', 
            f'{audio_tmpdir.name}/{audio_basename}_16kHz.wav', 
            f'{trans_tmpdir.name}/{audio_basename}.json'
            ])
    audio_tmpdir.cleanup()
    return trans_tmpdir

if __name__ == '__main__':
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
