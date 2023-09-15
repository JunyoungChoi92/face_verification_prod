<h2>얼굴 인증 과정 module의 reproduction 페이지입니다.</h2>

테스팅 환경 <br>
    MacBook Pro 16(m1 chip) Ventura 13.5.2(22G91)
    
가상환경 설정 <br>

    1. 상위 폴더에서, terminal에 conda env create --file environment.yaml를 입력합니다.
    2. conda activate fv2_reprod를 입력합니다.
    3. python interpreter 경로를 fv2_reprod로 변경합니다.

사용방법 <br>

1. Active Liveness detection <br>
    liveness_detection_by_face_rotation_task.py를 실행한 뒤, 브라우저로 127.0.0.1:8000 로 접근합니다. <br>

2. Face ROI 추출 및 검색 <br>
    a. dataset 폴더에 Validation을 위한 한국인 Face 데이터셋(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=161 중 Validation folder)을 다운로드 받아 넣습니다. <br>
    b. search_for_my_face.py를 실행한 뒤, 브라우저로 127.0.0.1:8000 로 접근합니다. <br>

