/*
Copyright 2017 Google Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

'use strict';

var videoElement = document.querySelector('video');
var videoSelect = document.querySelector('select#videoSource');
const screenshotButton = document.querySelector('#screenshot-button');
const canvas = document.createElement('canvas');
const moneyElem = document.getElementById('money');
const tmpImg = document.getElementById('tmpImg');

navigator.mediaDevices.enumerateDevices()
  .then(gotDevices).then(getStream).catch(handleError);

videoSelect.onchange = getStream;

function gotDevices(deviceInfos) {
  for (var i = 0; i !== deviceInfos.length; ++i) {
    var deviceInfo = deviceInfos[i];
    var option = document.createElement('option');
    option.value = deviceInfo.deviceId;
    if (deviceInfo.kind === 'videoinput') {
      option.text = deviceInfo.label || 'camera ' +
        (videoSelect.length + 1);
      videoSelect.appendChild(option);
    } else {
      console.log('Found one other kind of source/device: ', deviceInfo);
    }
  }
}

function getStream() {
  if (window.stream) {
    window.stream.getTracks().forEach(function(track) {
      track.stop();
    });
  }

  var constraints = {
    video: {
      deviceId: {exact: videoSelect.value},
      width: {ideal: screen.availHeight},
      height: {ideal: screen.availWidth},
    }
  };

  navigator.mediaDevices.getUserMedia(constraints).
    then(gotStream).catch(handleError);
}

function gotStream(stream) {
  window.stream = stream; // make stream available to console
  videoElement.srcObject = stream;
}

function handleError(error) {
  console.log('Error: ', error);
}

screenshotButton.onclick = function() {
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  canvas.getContext('2d').drawImage(videoElement, 0, 0);
  document.body.classList.add('predicting');
  let predictCleanup = () => {
    tmpImg.src = '';
    moneyElem.textContent = total.toString();

    document.body.classList.remove('predicting');
    document.body.classList.remove('ok');
  };
  canvas.toBlob(b => {
    let blobUrl = URL.createObjectURL(b);
    tmpImg.src = blobUrl;
    tmpImg.onload = function () {
      if(!blobUrl) return;
      URL.revokeObjectURL(blobUrl);
      blobUrl = null;
    };
    let fd = new FormData();
    fd.append('im', b);
    fetch('/predict', {
      method: 'POST',
      body: fd
    }).then(r => {
      let data = r.headers.get('data');
      if (data) {
        data = JSON.parse(data);
      }
      return r.blob().then(b => {
        return [b, data];
      });
    }).then(d => {
      let [blob, data] = d;
      // [{"label": "1", "score": 0.9972336888313293}, {"label": "1", "score": 0.9921825528144836}]
      let total = data.map(coin=>coin.label*1).reduce((a, b)=>a+b, 0);
      blobUrl = URL.createObjectURL(blob);
      tmpImg.src = blobUrl;
      setTimeout(()=>{
        moneyElem.textContent = total.toString();
        document.body.classList.add('ok');
        setTimeout(()=>{
          predictCleanup();
        }, 3000);
      }, 750);
    }).catch(e => {
      console.error(e);
      predictCleanup();
    });
  }, 'image/jpeg');
};

let fulled = false;
document.documentElement.addEventListener('click', function(e){
  if(fulled) return;
  document.documentElement.requestFullscreen();
  fulled = true;
});