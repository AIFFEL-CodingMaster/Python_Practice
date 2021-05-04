# 가상환경 생성
conda env create -f env.yaml
# 가상환경 실행
conda activate mini_project
# 패키지 설치
pip install -r requirements.txt
# django command
python manage.py migrate
# runserver
python manage.py runserver

