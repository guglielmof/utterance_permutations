import requests
import json
def annotate(string, lp=0.01):

    data = {'text': string, 'gcube-token':'18cd3cc9-cc01-492d-8cd7-e85b5c369ef5-843339462'}
    response = json.loads(requests.post('https://tagme.d4science.org/tagme/tag', data).text)['annotations']
    return [s['title'] if 'title' in s else s['spot'] for s in response if s['link_probability']>lp]