import json
import unittest
from string import Template

from lapps.discriminators import Uri

import app
import subprocess
from mmif import AnnotationTypes, DocumentTypes, Mmif, __specver__


class TestKaldiApp(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.kaldi_app = app.Kaldi()
        self.test_output = []
        app.TRANSCRIPT_DIR = 'test_output'
        completed_process = subprocess.run(
            [
                'clams',
                'source',
                'audio:/audio_in/app/gbh/wgbh-audios/cpb-aacip-259-154dqq73.h264.mp4.mp3',
                'audio:/audio_in/app/gbh/wgbh-audios/cpb-aacip-259-dj58gh9t.h264.mp4.mp3',
                'audio:/audio_in/app/gbh/wgbh-audios/cpb-aacip-259-f47gtp3c.h264.mp4.mp3'
            ],
            stdout=subprocess.PIPE
        )
        self.mmif_str = completed_process.stdout.decode('utf8')

    def test_appmetadata(self):
        metadata = json.loads(self.kaldi_app.appmetadata())
        with open('test_output/appmetadata.json') as target:
            target_metadata = json.loads(Template(target.read()).substitute(version=app.APP_VERSION,
                                                                            specver=__specver__,
                                                                            wrappee=app.WRAPPED_IMAGE))
        self.assertDictEqual(target_metadata, metadata)

    def test_annotate(self):
        self.kaldi_app.annotate(self.mmif_str, run_kaldi=False)

    @unittest.skip
    def test_in_place_annotate(self):
        string_annotate = self.kaldi_app.annotate(self.mmif_str, run_kaldi=False)
        mmif_obj = Mmif(self.mmif_str)
        self.kaldi_app.annotate(mmif_obj, run_kaldi=False)
        mmif_obj.freeze_views()
        self.assertEqual(Mmif(string_annotate), mmif_obj)


class TestAnnotation(unittest.TestCase):

    def setUp(self) -> None:
        self.kaldi_app = app.Kaldi()
        self.test_output = []
        app.TRANSCRIPT_DIR = 'test_output'
        completed_process = subprocess.run(
            [
                'clams',
                'source',
                'audio:/audio_in/app/gbh/wgbh-audios/cpb-aacip-259-154dqq73.h264.mp4.mp3',
                'audio:/audio_in/app/gbh/wgbh-audios/cpb-aacip-259-dj58gh9t.h264.mp4.mp3',
                'audio:/audio_in/app/gbh/wgbh-audios/cpb-aacip-259-f47gtp3c.h264.mp4.mp3'
            ],
            stdout=subprocess.PIPE
        )
        self.mmif_str = completed_process.stdout.decode('utf8')
        self.mmif_output = Mmif(self.kaldi_app.annotate(self.mmif_str, run_kaldi=False))

    def test_mmif_list_lengths(self):
        self.assertEqual(3, len(self.mmif_output.documents))
        self.assertEqual(3, len(self.mmif_output.views))
        self.assertEqual(4217, len(self.mmif_output['v_0'].annotations))
        self.assertEqual(13778, len(self.mmif_output['v_1'].annotations))
        self.assertEqual(12788, len(self.mmif_output['v_2'].annotations))

    def test_view_contents_counts(self):
        v_0 = self.mmif_output['v_0']
        v_1 = self.mmif_output['v_1']
        v_2 = self.mmif_output['v_2']

        self.assertEqual(1, len([anno for anno in v_0.annotations if anno.at_type == DocumentTypes.TextDocument.value]))
        self.assertEqual(1405, len([anno for anno in v_0.annotations if anno.at_type == Uri.TOKEN]))
        self.assertEqual(1405, len([anno for anno in v_0.annotations if anno.at_type == AnnotationTypes.TimeFrame.value]))
        self.assertEqual(1406, len([anno for anno in v_0.annotations if anno.at_type == AnnotationTypes.Alignment.value]))

        self.assertEqual(1, len([anno for anno in v_1.annotations if anno.at_type == DocumentTypes.TextDocument.value]))
        self.assertEqual(4592, len([anno for anno in v_1.annotations if anno.at_type == Uri.TOKEN]))
        self.assertEqual(4592, len([anno for anno in v_1.annotations if anno.at_type == AnnotationTypes.TimeFrame.value]))
        self.assertEqual(4593, len([anno for anno in v_1.annotations if anno.at_type == AnnotationTypes.Alignment.value]))

        self.assertEqual(1, len([anno for anno in v_2.annotations if anno.at_type == DocumentTypes.TextDocument.value]))
        self.assertEqual(4262, len([anno for anno in v_2.annotations if anno.at_type == Uri.TOKEN]))
        self.assertEqual(4262, len([anno for anno in v_2.annotations if anno.at_type == AnnotationTypes.TimeFrame.value]))
        self.assertEqual(4263, len([anno for anno in v_2.annotations if anno.at_type == AnnotationTypes.Alignment.value]))

    def test_contains(self):
        self.assertDictEqual(
            {
                DocumentTypes.TextDocument.value: {},
                Uri.TOKEN: {},
                AnnotationTypes.TimeFrame.value: {'unit': 'milliseconds', 'document': 'd1'},
                AnnotationTypes.Alignment.value: {}
            },
            json.loads(self.mmif_output['v_0'].metadata.contains.serialize())
        )

    def test_produces_contract_actual(self):
        produces: set = set(json.loads(self.kaldi_app.appmetadata())['produces'])
        actually_produces: set = set(str(anno.at_type) for anno in self.mmif_output['v_0'].annotations) \
            | set(str(anno.at_type) for anno in self.mmif_output['v_1'].annotations) \
            | set(str(anno.at_type) for anno in self.mmif_output['v_2'].annotations)
        self.assertSetEqual(produces, actually_produces)

    def test_produces_contract_contains(self):
        produces: set = set(json.loads(self.kaldi_app.appmetadata())['produces'])
        contains_produces: set = set(self.mmif_output['v_0'].metadata.contains.keys()) \
            | set(self.mmif_output['v_1'].metadata.contains.keys()) \
            | set(self.mmif_output['v_2'].metadata.contains.keys())
        self.assertSetEqual(produces, contains_produces)


if __name__ == '__main__':
    unittest.main()
