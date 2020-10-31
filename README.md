# Database for Direction-of-Voice (DoV) Estimation
This is the research repository for Direction-of-Voice (DoV) Estimation for Intuitive Speech Interaction with Smart Devices Ecosystems (UIST 2020). It contains the the database. [More details can be found here](https://karan-ahuja.com/dov.html).

Link to the dataset: https://www.dropbox.com/s/gtw7o0nj0h7j4gy/subjectrecording.zip

## Dataset Description

<img align="right" width="325" height="300" src="https://karan-ahuja.com/assets/docs/paper/DoVDataCollection.png">

The data is organized in the following manner:

* 10 participants (s1 to s10)
* 2 utterances (recording0 and recording1)
* 2 sessions (trial1 and trial2)
* 2 rooms (upstairs and downstairs)
* 2 device placements (wall and nowall)
* 3 user distances (1m, 3m and 5m)
* 3 polar positions (X0, X1 and X2)
* 8 angles (DoV Angle: 45 degree increments from 0 to 360 degrees)

This leads to: 10 × 2 × 2 × 2 × 2 × 3 × 3 × 8 = 11520 recordings

The hardware used is a Seeed ReSpeaker [USB Mic Array](https://www.seeedstudio.com/ReSpeaker-USB-Mic-Array-p-4247.html) ([wiki here](https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/)) flashed with the 6 channel, 48kHz sampling frequency (specified as "48k_6_channels_firmware.bin". Here channel 0 is processed audio for ASR, channel 1-4 are the 4 microphones' raw data and channel 5 is playback.

The data is organized as follows:

* ParticipantID/
  * ParticipantID_RoomID_DevicePlacementID_SessionID/
    * PolarPositionID_Distance_PolarAngle/
      * UtteranceID_DoVAngle_MicChannel
  



## Reference
Karan Ahuja, Andy Kong, Mayank Goel, and Chris Harrison. 2020. Direction-of-Voice (DoV) Estimation for Intuitive Speech Interaction with Smart Devices Ecosystems. In Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology (UIST '20). Association for Computing Machinery, New York, NY, USA, 1121–1131. DOI:https://doi.org/10.1145/3379337.3415588

[Download the paper here](https://karan-ahuja.com/assets/docs/paper/dov.pdf).

BibTex Reference:
```
@inproceedings{10.1145/3379337.3415588,
author = {Ahuja, Karan and Kong, Andy and Goel, Mayank and Harrison, Chris},
title = {Direction-of-Voice (DoV) Estimation for Intuitive Speech Interaction with Smart Devices Ecosystems},
year = {2020},
isbn = {9781450375146},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3379337.3415588},
doi = {10.1145/3379337.3415588},
booktitle = {Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology},
pages = {1121–1131},
numpages = {11},
keywords = {addressability, speaker orientation, voice interfaces},
location = {Virtual Event, USA},
series = {UIST '20}
}
```


## Disclaimer

```
THE PROGRAM IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW THE AUTHOR WILL BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```
