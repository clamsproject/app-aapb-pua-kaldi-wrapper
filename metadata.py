"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""
from lapps.discriminators import Uri
from mmif import DocumentTypes, AnnotationTypes

from clams.appmetadata import AppMetadata

timeunit = 'milliseconds'


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """

    metadata = AppMetadata(
        name="AAPB-PUA Kaldi Wrapper",
        description="A CLAMS wrapper for Kaldi-based ASR software originally developed by PopUpArchive and hipstas, "
                    "and later updated by Kyeongmin Rim at Brandeis University. Wrapped software can be "
                    "found at https://github.com/brandeis-llc/aapb-pua-kaldi-docker . ",
        app_license="Apache 2.0",
        identifier=f"aapb-pua-kaldi-wrapper",
        url="https://github.com/clamsproject/app-aapb-pua-kaldi-wrapper",
        analyzer_version="v4",
        analyzer_license="UNKNOWN",
    )
    metadata.add_input(DocumentTypes.AudioDocument)
    metadata.add_output(DocumentTypes.TextDocument)
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit=timeunit)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(Uri.TOKEN)
    metadata.add_parameter(name="use_speech_segmentation",
                           type="boolean",
                           description="When true, the app looks for existing TimeFrame { \"frameType\": "
                                       "\"speech\" } annotations, and runs ASR only on those frames, instead of "
                                       "entire audio files.",
                           default="true")
    return metadata

# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    sys.stdout.write(appmetadata().jsonify(pretty=True))
