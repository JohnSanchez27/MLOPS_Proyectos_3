from sqlalchemy import create_engine


PASSWORD = 'Compaq*87'

#engine_raw_data = create_engine(f'mysql+pymysql://root:{PASSWORD}@localhost:3306/RAW_DATA')
#engine_clean_data = create_engine(f'mysql+pymysql://root:{PASSWORD}@localhost:3306/CLEAN_DATA')

engine_raw_data = create_engine(f'mysql+pymysql://root:{PASSWORD}@localhost:8082/RAW_DATA')
engine_clean_data = create_engine(f'mysql+pymysql://root:{PASSWORD}@localhost:8082/CLEAN_DATA')
