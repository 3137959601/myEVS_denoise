import csv
p = r"data/ED24/myPedestrain_06/EBF/roc_ebf_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv"
rows = 0
non = 0
tags = set()
non_tags = set()
with open(p, 'r', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        rows += 1
        t = row.get('tag', '')
        tags.add(t)
        v = (row.get('esr_mean') or '').strip()
        if v:
            non += 1
            non_tags.add(t)
print('rows', rows, 'tags', len(tags), 'nonempty_esr_rows', non, 'tags_with_esr', len(non_tags))
