import json
import os
import shutil
import subprocess
from typing import Dict

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes, Text
from mmif.serialize.model import MmifObjectEncoder

KALDI_MEDIA_DIRECTORY = '/audio_in'
KALDI_16KHZ_DIRECTORY = '/audio_in_16khz'
KALDI_EXPERIMENT_DIR = '/kaldi/egs/american-archive-kaldi/sample_experiment'
KALDI_OUTPUT_DIR = os.path.join(KALDI_EXPERIMENT_DIR, 'output')
KALDI_VERSION = 'IDK'
TEXT_DOCUMENT_PREFIX = 'td'
TIME_FRAME_PREFIX = 'tf'
ALIGNMENT_PREFIX = 'a'


class Kaldi(ClamsApp):

    def __init__(self):
        self.metadata = {
            "name": "Kaldi Wrapper",
            "description": "This tool wraps the Kaldi ASR tool",
            "vendor": "Team CLAMS",
            "iri": f"http://mmif.clams.ai/apps/kaldi/{KALDI_VERSION}",
            "requires": [DocumentTypes.AudioDocument],
            "produces": [DocumentTypes.TextDocument, AnnotationTypes.TimeFrame, AnnotationTypes.Alignment]
        }
        super().__init__()

    def appmetadata(self):
        return json.dumps(self.metadata, cls=MmifObjectEncoder)

    def sniff(self, mmif) -> bool:
        if type(mmif) is not Mmif:
            mmif = Mmif(mmif)
        return len(mmif.get_documents_locations(DocumentTypes.AudioDocument.value)) > 0

    def annotate(self, mmif, run_kaldi=True) -> Mmif:
        if type(mmif) is not Mmif:
            mmif = Mmif(mmif)

        # get AudioDocuments with locations
        docs = [document for document in mmif.documents
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
            view: View = mmif.new_view()
            self.stamp_view(view, docs_dict[basename].id)
            # make annotations
            for index, word_obj in enumerate(transcript['words']):
                td = self.create_td(word_obj['word'], index)
                tf = self.create_tf(word_obj['time'], word_obj['duration'], index)
                align = self.create_align(td, tf, index)
                view.add_document(td)
                view.add_annotation(tf)
                view.add_annotation(align)

        return mmif

    @staticmethod
    def create_td(word: str, index: int) -> Document:
        text = Text()
        text.value = word
        td = Document()
        td.at_type = DocumentTypes.TextDocument
        td.id = TEXT_DOCUMENT_PREFIX + str(index + 1)
        td.properties.text = text
        return td

    @staticmethod
    def create_tf(time: float, duration: str, index: int) -> Annotation:
        tf = Annotation()
        tf.at_type = AnnotationTypes.TimeFrame
        tf.id = TIME_FRAME_PREFIX + str(index + 1)
        tf.properties['frameType'] = 'speech'
        # times should be in milliseconds
        tf.properties['start'] = int(time * 1000)
        tf.properties['end'] = int((time + float(duration)) * 1000)
        return tf

    @staticmethod
    def create_align(td: Document, tf: Annotation, index: int) -> Annotation:
        align = Annotation()
        align.at_type = AnnotationTypes.Alignment
        align.id = ALIGNMENT_PREFIX + str(index + 1)
        align.properties['source'] = tf.id
        align.properties['target'] = td.id
        return align

    def stamp_view(self, view: View, tf_source_id: str) -> None:
        if view.is_frozen():
            raise ValueError("can't modify an old view")
        view.metadata['app'] = self.metadata['iri']
        view.new_contain(DocumentTypes.TextDocument)
        view.new_contain(AnnotationTypes.TimeFrame, {'unit': 'milliseconds', 'document': tf_source_id})
        view.new_contain(AnnotationTypes.Alignment)


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
    kaldi_app = Kaldi()
    kaldi_service = Restifier(kaldi_app)
    kaldi_service.run()
