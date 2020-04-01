#### skeleton-data-parser



##### 자동 skeleton parser

skeleton_extractor

자동으로 원본 동영상에서 추출할 구간을 설정하면 해당 구간의 영상을 4FPS의 프레임 이미지로 제작하고 
이를 모델을 돌려서 벡터값을 추출한다. 

```
./skeleton_extractor.sh s1 s2 s3
s1 : 시작 시간 (ex. 00:00:21.0)
s2 : 비디오 이름 (ex. door)
s3 : 재생 시간 (ex. 00:00:10.0)

ex ) ./skeleton_extractor.sh 00:00:21.0 door 00:00:10.0
```
