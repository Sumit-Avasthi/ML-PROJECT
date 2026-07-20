import logging
import os
from datetime import datetime


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  ## Naming convention of Log file
log_path = os.path.join(os.getcwd(),"logs")

os.makedirs(log_path,exist_ok=True) #Even though there is file keep append on that

log_file_path=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)

