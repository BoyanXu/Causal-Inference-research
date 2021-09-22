# sed "s|'all_awardings': .* 'created_utc'|'created_utc'|g" test.txt
# sed "s|'domain': .* 'title'|'title'|" test.txt
# sed "s|, 'total_awards_received': .* 'wls': [0-9]+\}||g" test.txt
# sed -E "s|, 'total_awards_received': .* 'wls': [0-9]+||g" test.txt

sed -E -e "s|'all_awardings': .* 'created_utc'|'created_utc'|g" -e "s|'domain': .* 'title'|'title'|"  -e "s|, 'total_awards_received': .* 'wls': [0-9]+\}|}|g" -e "s|'|\"|g" test.txt > output.json