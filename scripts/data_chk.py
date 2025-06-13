import pandas as pd
import os
import numpy as np

dataset_base_path = '/virtual_lab/rlwrld/david/pi_0/openpi/data/rlwrld_gr1/gr1-cube-dataset/data/chunk-000/'

# 다른 파일들을 검사
# files_to_check = ['episode_000001.parquet', 'episode_000010.parquet', 'episode_000050.parquet'] # 몇 개 선택
files_to_check = ['episode_000000.parquet', 'episode_000050.parquet']

for file_name in files_to_check:
    full_path = os.path.join(dataset_base_path, file_name)
    print(f"\n--- Checking file: {file_name} ---")
    try:
        df = pd.read_parquet(full_path)
        # 이전 data_chk.py 코드의 'DataFrame Info' 부분과 'Inspecting specific object columns' 부분을 여기에 붙여넣어 실행합니다.
        # 특히 df.info()에서 여전히 'object' 타입으로 나타나는 컬럼들을 다시 확인하고,
        # 해당 컬럼들에 대해 'isinstance(x, (str))' 체크를 수행하여 문자열이 있는지 찾아보세요.

        # 예시 (이전 코드에서 가져온 부분):
        print("\nDataFrame Info (컬럼별 타입 정보):\n")
        df.info()

        columns_to_check = ['action', 'observation.state', 'object.state'] # 기타 object 컬럼도 추가
        for col_name in columns_to_check:
            if col_name in df.columns and df[col_name].dtype == 'object':
                print(f"\nChecking column: '{col_name}'")
                first_element = df[col_name].iloc[0]
                print(f"  Type of first element in column: {type(first_element)}")
                if hasattr(first_element, '__iter__') and not isinstance(first_element, (str, bytes)):
                    if len(first_element) > 0:
                        print(f"  Type of first sub-element: {type(first_element[0])}")
                    problematic_rows = []
                    for idx, cell_value in enumerate(df[col_name]):
                        if not hasattr(cell_value, '__iter__') or isinstance(cell_value, (str, bytes)):
                            problematic_rows.append((idx, f"Not an iterable or is a string: {cell_value}"))
                        else:
                            if not all(isinstance(x, (int, float, np.number, type(None))) or (isinstance(x, (str)) and (x.lower() == 'nan' or x == '')) for x in cell_value):
                                problematic_sub_elements = [x for x in cell_value if not (isinstance(x, (int, float, np.number, type(None))) or (isinstance(x, (str)) and (x.lower() == 'nan' or x == '')))]
                                if problematic_sub_elements:
                                    problematic_rows.append((idx, f"Non-numeric elements found: {problematic_sub_elements}"))
                        if len(problematic_rows) > 5:
                            break
                    if problematic_rows:
                        print(f"  Found {len(problematic_rows)} potential problematic entries in '{col_name}'. Showing first 5:")
                        for row_idx, msg in problematic_rows[:5]:
                            print(f"    Row {row_idx}: {msg}. Full value: {df[col_name].iloc[row_idx]}")
                    else:
                        print(f"  No obvious non-numeric elements found in '{col_name}' lists/arrays.")
                else:
                    print(f"  '{col_name}' contains direct non-iterable or string values. Sample: {first_element}")

    except FileNotFoundError:
        print(f"Error: File not found at {full_path}")
    except Exception as e:
        print(f"An unexpected error occurred for {file_name}: {e}")
