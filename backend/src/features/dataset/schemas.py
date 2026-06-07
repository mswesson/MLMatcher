from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    row_count: int
