folders=("n03085013" "n02783161" "n06255613" "n03085219" "n03793489" "n04380346" "n03960490" "n04284002" "n03880531" "n02880940" "n07739506" "n07767847" "n07753592" "n07745940" "n07753275" "n07730207" "n07710616" "n07723559" "n12997919" "n07722217" "n07715103" "n07735510" "n02084071" "n02121808" "n01503061" "n02512053" "n02439033" "n02200198" "n02958343" "n02834778" "n02924116" "n04225987" "n04389033" "n04208936" "n02808440" "n02808304" "n04254009" "n03475581" "n04453156" "n03236735" "n03688605" "n03497657" "n04254777" "n04200000" "n04266375" "n03481172" "n04154340" "n03995372" "n02883344" "n04154565" "n13104059" "n11669921" "n12102133" "n09247410" "n09217230"  "n11691857" "n09618957" "n03378442" "n01900150" "n13865904" "n03544360" "n03613294" "n04256520" "n04041069" "n03151077" "n03467517" "n03452741" "n02803666" "n03759954" "n04043733" "n02818832" "n03938244" "n02694662" "n04550184" "n13881644" "n06278475" "n02870526" "n02778669" "n03262248" "n04028315" "n02756977" "n13863020" "n04118776" "n03483823" "n02992795" "n02846511" "n04272054" "n04981658" "n03814906" "n02774152" "n03476313" "n07873807" "n07679356" "n07577144" "n07700003" "n07697100" "n03139464" "n02951703" "n03954393" "n04306847" "n01704323" "n00478262" "n00482298" "n02882301" "n02802426" "n00440747" "n02782093" "n04507155" "n03028079" "n02942699" "n02898711")

for folder in "${folders[@]}"; do
    echo $folder
    gsutil cp gs://disdat-archive/imnet/$folder.tar ../data-110
    mkdir -p ../data-110/$folder
    tar -xf ../data-110/$folder.tar -C ../data-110/$folder
    rm ../data-110/$folder.tar
done