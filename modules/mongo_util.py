import requests

class VideoTags:
    def __init__(self, base_url):
        self.base_url = base_url
        
    #### Get tags
    def get(self, blob_name, bucket):
        url = f'{self.base_url}/tags/{blob_name}?b={bucket}'
        res = requests.get(url)
        if not res.ok:
            data = None
        else: data = res.json()
        return {'status': res.ok, 'status_code': res.status_code, 'message': res.reason, 'data': data}
    
    #### Post tags
    def post(self, tags, blob_name, bucket):
        url_tags = f'{self.base_url}/tags';
        data = {
            'tags': tags,
            'b': bucket,
            'blob_name': blob_name,
        };
        res = requests.post(url_tags, json=data)
        if not res.ok:
            data = None
        else: data = res.json()
        return {'status': res.ok, 'status_code': res.status_code, 'message': res.reason, 'data': data}
    
    #### Delete tag
    def delete(self, tag, blob_name, bucket):
        url = f'{self.base_url}/tags/{tag}?blob_name={blob_name}&b={bucket}'
        res = requests.delete(url)
        if not res.ok:
            data = None
        else: data = res.json()
        return {'status': res.ok, 'status_code': res.status_code, 'message': res.reason, 'data': data}
