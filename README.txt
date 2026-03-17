



https://www.libreoffice.org/donate/dl/win-x86_64/26.2.1/ko/LibreOffice_26.2.1_Win_x86-64.msi

pip install pdf2image Pillow

https://github.com/oschwartz10612/poppler-windows/releases




# 기본 실행 (PDF + 페이지 JPG + 내부 이미지 모두)
python docx_converter.py 입사지원서_코리아써치_.docx

# 해상도 높이기 (기본 150 DPI)
python docx_converter.py test.docx --dpi 300

# 페이지 JPG만 건너뛰기
python docx_converter.py 입사지원서_코리아써치_.docx --no-jpg

git init
git remote add origin https://github.com/jungskky/knowledge.git

git add .
git commit -m "초기 커밋"

# main으로 push
git push -u origin main

git init
git add .
git commit -m "first commit"
git branch -m master        # 브랜치명 master로 설정
git remote add origin https://github.com/jungskky/knowledge.git
git push -u origin master