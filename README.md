# AicupOCR (初稿)
## Info
- 將定位model與字串辨識model合起來，使得使用者直接輸入一張或多張圖片，則得圖片繁體中文字位置與內容。(佔限繁體中文)
- 定位模型使用 [DB github code](https://github.com/MhLiao/DB)
- 辨識模型使用 [CRNN github code](https://github.com/clovaai/deep-text-recognition-benchmark) 
## Ｓtart
- 將model 資料夾放入root  資料夾中
- 用aicup.tar 建立新的docker image 和 container
## Run
~~~bash= python!
CUDA_VISIBLE_DEVICES=1 python aicupOCR.py --rgb --output_file_name name/of/result/txt --result_dir result --image_dir_path ./path/to/image/directory
--threshold_crnn 0.1 --visualize
~~~
- `threshold_crnn` : in range [0,1]
- `--rgb` : if DB is trained by 3 channel(rgd)
- `--visualize`: 是否需要visualize
- visualization output will generate in result_dir/img
- output_file_name 將會產生再 result_dir 中
### 支援AICUP2
- 加上 aicup2_file 就行，沒加就會跑 Detection
~~~bash= python!
CUDA_VISIBLE_DEVICES=1 python aicupOCR.py --rgb --aicup2_file path/to/GT/file
~~~
- aicup2_file 
    - EX: ./AdvancedComp_public_test/easy/coordinates_easy.txt
