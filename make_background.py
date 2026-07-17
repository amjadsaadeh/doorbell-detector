import os
import json
from label_studio_sdk.client import LabelStudio
import tqdm

from label_studio_sdk.data_manager import Filters, Column, Type, Operator

if __name__ == '__main__':

    url = os.getenv('LABEL_STUDIO_URL')
    api_key = os.getenv('LABEL_STUDIO_API_KEY')

    client = LabelStudio(
        base_url=url,
        api_key=api_key,
    )

    
    filters = Filters.create(Filters.AND, [
        Filters.item(
            Column.total_annotations,
            Operator.EQUAL,
            Type.Number,
            Filters.value(0)
        )
    ])

    
    response = client.tasks.list(project=1, query=json.dumps({'filters': filters}))

    for item in tqdm.tqdm(response):
        client.annotations.create(id=item.id, result=[
            {
                'type': 'labels',
                'value': {
                    'labels': ['background'],
                    'start': 0,
                    'end': 7,
                    'channel': 0
                },
                'from_name': 'label',
                'to_name': 'audio',
            }
        ])
    # # alternatively, you can paginate page-by-page
    # for page in response.iter_pages():
    #     yield page
