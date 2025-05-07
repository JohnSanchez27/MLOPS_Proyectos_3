from sqlalchemy import create_engine

#engine = create_engine('mysql+pymysql://root:airflow@10.43.101.156:8082/RAW_DATA')
#engine_2 = create_engine('mysql+pymysql://root:airflow@10.43.101.156:8082/CLEAN_DATA')
PASSWORD = 'Compaq*87'

engine_raw_data = create_engine(f'mysql+pymysql://root:{PASSWORD}@localhost:3306/RAW_DATA')
engine_clean_data = create_engine(f'mysql+pymysql://root:{PASSWORD}@localhost:3306/CLEAN_DATA')