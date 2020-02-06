from torch.utils.data import DataLoader, Dataset
import torch
import time
import datetime
import torch.multiprocessing as mp
num_batches = 110

print("File init")

class DataClass:
    def __init__(self, x):
        self.x = x


class SleepDataset(Dataset):
    def __len__(self):
        return num_batches

    def __getitem__(self, idx):
        info = torch.utils.data.get_worker_info()
        print(f"sleep on {idx} id={info.id}")
        time.sleep(5)
        print(f"finish sleep on {idx} at {datetime.datetime.now()}")
        return DataClass(torch.randn(5))


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


def _set_seed(worker_id):
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed(worker_id)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    num_workers = mp.cpu_count() - 1
    print(f"num of workers {num_workers}")
    dataset = SleepDataset()

    print("haha")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_set_seed,
        collate_fn=collate_fn,
    )

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched.x)
    dataloader = iter(dataloader)
    for i in range(1000):
        print(next(dataloader).x)
