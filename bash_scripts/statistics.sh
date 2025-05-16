num_files=$(find .gocomics_downloads -type f | wc -l)
num_dirs=$(find ./gocomics_downloads -type d | wc -l)
echo "[comic] num_comics: $num_files"
echo "[comic] num_creators: $num_dirs"



num_files=$(find .gocomics_downloads_political -type f | wc -l)
num_dirs=$(find .gocomics_downloads_political -type d | wc -l)
echo "[political] num_comics: $num_files"
echo "[political] num_creators: $num_dirs"