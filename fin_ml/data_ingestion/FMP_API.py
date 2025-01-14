import pandas as pd
from fin_ml.utils.data_ingestion import make_request



BASE_URL = "https://financialmodelingprep.com/api"



def get_individual_series(
        api_key: str, output_path: str, api_params: tuple
) -> None:
    """
    """
    for input_data in api_params:
        dataset_name, api_version, dataset_params, output_type = input_data
        
        print(f"Requesting data for '{dataset_name}'...")
        response = make_request(
            method='GET',
            url=f"{BASE_URL}/{api_version}/{dataset_name}",
            params={"apikey": api_key, **dataset_params}
        )
        print('Response received.')
        
        file_path = output_path + f'{dataset_name}.csv'
        
        if output_type == 'json':
            output_data = pd.DataFrame(response.json())
            output_data.to_csv(file_path, index=False)
        elif output_type == 'csv_bytes':
            output_data = response.content
            with open(file_path, 'wb') as f:
                f.write(output_data)
            
        print(f'Content exported (len={len(output_data)})')