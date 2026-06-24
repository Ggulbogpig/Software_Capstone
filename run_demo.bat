@echo off
call C:\Users\hbsss\anaconda3\Scripts\activate.bat openad

cd /d D:\Software_Capstone\EmpartACD\empart

python pipeline_local.py

pause