This is a RUNPOD repackaging project, and not a code rebuilding project.  The code is known to work, it is about containerizing the project so that is can run as a Runpod Container.
use context7 to familiarize your self with best practices for building runpod container applications.
Use fetch7 to read https://github.com/Francis-Rings/StableAvatar and familiarize yourself with the project

0) Use base image runpod/pytorch:0.7.0-cu1263-torch260-ubuntu2204

**Everything below here happens at **RUNTIME** and is not part of the container.  The working directory is /workspace

0) install ffmpeg from OS packages
1) Python Stream buffer set to 1
2) Create Python 3.11 virtual environment.  Taking care to understand where modules are getting installed, and where programs are being run from.
3) Clone repo https://github.com/Francis-Rings/StableAvatar.git
4) Change to StableAvatar directory
4) pip install -r requirements.txt
6) Install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
7) pip install "huggingface_hub[cli]"
8) mkdir checkpoints
9) huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
10) pip install audio-separator
11) The project is for Runpod **ABSOLUTELY** do not install python modules on the localhost or try to build the container on the localhost.  This is only to be built on github and ran from RUNPOD
12) Always use SSH when interacting with github.  SSH keys have already been configured
13) Change the origin of the project to be sruckh/StableAvatar
14) Set up github action to build and deploy to Dockerhub.  The secrets DOCKER_USERNAME and DOCKER_PASSWORD are configured for pushing to gemneye/ dockerhub repository.

A gradio interface will need to be coded to support all of the functionality of inference.sh
Additionally there should be a tab for extracting audio from a video file by using python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav".  The output audio should be able to be used as audio input on the first tab that provides the inference.sh functionality.
There should also be a Voice seperation tab the supports the functionality, python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav".  Again the output should able to be used as input on the first tab that provides the inference.sh fucntionality.

Include this text below the prompt input box the give the user instruction on the format of the prompt:  [Description of first frame]-[Description of human behavior]-[Description of background (optional)]

Have an option to merge the source audio with the output video that the inference.sh script creates.

Use the following ffmpeg command to merge the source audio with the output video
ffmpeg -i <input video> -i <input audio> -c:v libx265 -pix_fmt yuv420p -vf "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=out_color_matrix=bt709" -maxrate:v 3500k -bufsize 6000k -color_primaries bt709 -color_trc bt709 -colorspace bt709 -threads 0 -crf 22 -bf 2 -g 15 -movflags +faststart -c:a aac -profile:a aac_low -b:a 128k -map 0:v:0 -map 1:a:0 <output.mp4>

The filenames inside angle brackes <> should be based on the files configured and created earlier when running inference.

The final output video should be able to be downloaded.
