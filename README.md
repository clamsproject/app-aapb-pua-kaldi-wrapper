# app-puakaldi-wrapper
Wrapping a Kaldi instance (https://github.com/brandeis-llc/aapb-pua-kaldi-docker) to a CLAMS app. 

Requirements:

- Docker to run the code as a server in a Docker container
- curl or some other utility to send of an HTTP request to the server
- Python 3 with the `clams-python` module installed to create the MMIF input

You probably need to change the Docker resources settings since the memory default is too low for running the code, it should be updated to at least 6GB, but more is better. 

The simplest way to run this is to first pull two images, one with the Kaldi and Pop Up Archive models and one with the CLAMS wrapper (the latter is built on top of the former, so just the second pull should suffice):

```bash
$ docker pull brandeisllc/aapb-pua-kaldi:v4
$ docker pull keighrim/app-aapb-pua-kaldi-wrapper:v0.2.4
```

Now start the container:

```bash
$ docker run -p 5000:5000 -v /Users/Shared/archive:/data --rm keighrim/app-aapb-pua-kaldi-wrapper:v0.2.3
```

This assumes that we have a local directory `/Users/Shared/archive` which is mapped to the `\data` directory on the container. If that local directory has an audio file `audio/newshour-99-04-27-short.wav` then we can create an input MMIF file:

```bash
$ clams source audio:/data/audio/newshour-99-04-27-short.wav > input.mmif
```
(Make sure you installed the same `clams-python` package version specified in the [`requirements.txt`](requirements.txt).)

Call the service:

```bash
$ curl http://0.0.0.0:5000?pretty
$ curl -H "Accept: application/json" -X POST -d@input.mmif -s http://0.0.0.0:5000?pretty > output.mmif
```

The second command can take a while, expect the app to use at least as much time to run as the length of the audio file, probably longer.

