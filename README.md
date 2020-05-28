## VIDEO FACE DETECTION

Detect faces of the video every 1 second so this program will not brick your device. 

## REQUIREMENTS
* python 3 
* pip 3

## HOW TO USE
```
pip install -r requirements
python app.py 
```
_Currently this program doesn't ready for select video source or camera so you need to change source manually at_ 
`app.py` 

```
...

if __name__ == "__main__":
    video = App(path=???, min_confidence=0.5)
    video.start()
```
_replace ??? with path of your video or 0 to start an camera_

### ROAD MAP
* [x] detect faces that appears in video 
* [x] extract detected faces into some folder
* [ ] training data (grouping same people), with supervised learning so,
* [ ] confirm that is same people
* [ ] we need to naming grouped people
* [ ] support gui with feature select video source including camera