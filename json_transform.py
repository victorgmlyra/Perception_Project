import json

def json_transform(name):
    f = open('data/dataset/' + name + '.json')
    data = json.load(f)

    new_json = {}
    for img in data['labels']:
        w = img['metadata']['image']['width']
        h = img['metadata']['image']['height']
        rects, labels = [], []
        for key, value in img['annotations'].items():
            for r in value:
                min_x, min_y = r['data']['min']
                max_x, max_y = r['data']['max']
                rects.append([int(min_x*w), int(min_y*h), int(max_x*w), int(max_y*h)])
                labels.append(key)
        
        new_json[img['path']] = {'rects': rects, 'labels': labels}

    with open('data/dataset/' + name + '_torch.json', 'w') as outfile:
        json.dump(new_json, outfile)
