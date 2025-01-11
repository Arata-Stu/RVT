## 引数でgen1 gen4を指定
## gen1 gen4以外はassert

## 確認
if [ $# -ne 1 ]; then
    echo "Usage: $0 [gen1|gen4]"
    exit 1
fi

## loop
dt=("5" "10" "20" "50" "100")

# dt=("100")

## dtに応じてループ
for d in "${dt[@]}"; do
    echo $d
    ## gen1 gen4, dtに応じてファイル名を変更して実行
    python3 create_video.py -y ./config/"$1"_dt_"$d".yaml -o "$1"_dt_"$d".mp4 -m 3 -n 4
    
done


python3 create_video.py -y ./config/gen1_dt_5.yaml -o "$1"_GT.mp4 -m 2 -n 4

