import json
import os
import shutil
import subprocess
from typing import Dict, Iterator, Tuple, List, Union
import argparse

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes, Text
from lapps.discriminators import Uri

KALDI_MEDIA_DIRECTORY = '/audio_in'
KALDI_16KHZ_DIRECTORY = '/audio_in_16khz'
KALDI_EXPERIMENT_DIR = '/kaldi/egs/american-archive-kaldi/sample_experiment'
KALDI_OUTPUT_DIR = os.path.join(KALDI_EXPERIMENT_DIR, 'output')
KALDI_VERSION = 'IDK'
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
            "iri": f"http://mmif.clams.ai/apps/kaldi/{KALDI_VERSION}",
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

    def annotate(self, mmif: Union[str, dict, Mmif], run_kaldi=True) -> Mmif:
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
            # run setup
            setup(files)

        # get Kaldi's output
        json_transcripts: Dict[str, dict] = {}
        for transcript in os.listdir(os.path.join(KALDI_OUTPUT_DIR, 'json')):
            with open(os.path.join(KALDI_OUTPUT_DIR, 'json', transcript), encoding='utf8') as json_file:
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
            indices, doc = self.index_and_join_tokens(token['word'] for token in transcript['words'])
            # make annotations
            td = self.create_td(doc)
            view.add_document(td)
            for index, word_obj in enumerate(transcript['words']):
                tf = self.create_tf(word_obj['time'], word_obj['duration'], index)
                token = self.create_token(word_obj['word'], index, indices, f'{view.id}:{td.id}')
                align = self.create_align(tf, token, index)
                view.add_annotation(token)
                view.add_annotation(tf)
                view.add_annotation(align)

        return mmif_obj

    @staticmethod
    def index_and_join_tokens(tokens: Iterator[str]) -> Tuple[List[Tuple[int, int]], str]:
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
    def create_td(doc: str) -> Document:
        text = Text()
        text.value = doc
        td = Document()
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
    def create_align(tf: Annotation, token: Annotation, index: int) -> Annotation:
        align = Annotation()
        align.at_type = AnnotationTypes.Alignment.value
        align.id = ALIGNMENT_PREFIX + str(index + 1)
        align.properties['source'] = tf.id
        align.properties['target'] = token.id
        return align

    def stamp_view(self, view: View, tf_source_id: str) -> None:
        if view.is_frozen():
            raise ValueError("can't modify an old view")
        view.metadata['app'] = self.metadata['iri']
        view.new_contain(DocumentTypes.TextDocument.value)
        view.new_contain(Uri.TOKEN)
        view.new_contain(AnnotationTypes.TimeFrame.value, {'unit': 'milliseconds'})
        view.new_contain(AnnotationTypes.Alignment.value)


def setup(files: list) -> None:
    links = [os.path.join(KALDI_MEDIA_DIRECTORY, os.path.basename(file)) for file in files]

    # make 16khz directory
    os.mkdir(KALDI_16KHZ_DIRECTORY)

    # symlink these files to KALDI_MEDIA_DIRECTORY
    for file, link in zip(files, links):
        shutil.copy(file, link)
        clipped_name = link[:-4]
        subprocess.run(['ffmpeg', '-i', link, '-ac', '1', '-ar', '16000',
                         f'{clipped_name}_16kHz.wav'])
        subprocess.run(['mv', f'{clipped_name}_16kHz.wav', KALDI_16KHZ_DIRECTORY])

    subprocess.run([
        'python', '/kaldi/egs/american-archive-kaldi/run_kaldi.py',  # this is a Python 2 call
        KALDI_EXPERIMENT_DIR, KALDI_16KHZ_DIRECTORY])
    subprocess.run([
        'rsync', '-a', '/kaldi/egs/american-archive-kaldi/sample_experiment/output/', '/audio_in/transcripts/'
    ])
    subprocess.run(['rm', '-r', KALDI_16KHZ_DIRECTORY])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once',
                        type=str,
                        help='Use this flag if you want to run Kaldi on a path you specify, instead of running '
                             'the Flask app.')
    parser.add_argument('--no-kaldi',
                        help='Add this flag if Kaldi has already been run and you just want to re-annotate.')

    args = parser.parse_args()

    if args.once:
        with open('gbh/mmif.json') as mmif_in:
            mmif_str = mmif_in.read()

        kaldi_app = Kaldi()

        mmif_out = kaldi_app.annotate(mmif_str, run_kaldi=args.no_kaldi)
        with open('mmif_out.json', 'w') as out_file:
            out_file.write(mmif_out.serialize(pretty=True))

    else:
        kaldi_app = Kaldi()
        kaldi_service = Restifier(kaldi_app)
        kaldi_service.run()