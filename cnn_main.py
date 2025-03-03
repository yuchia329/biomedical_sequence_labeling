# cnn_main.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cnn_load_data import load_data, createTagSets
from cnn_model import BiLSTM_CRF

START_TAG = "<START>"
STOP_TAG  = "<STOP>"
PAD_TAG   = "<PAD>"
UNK_TAG   = "<UNK>"

def collate_fn(batch):
    """
    batch is a list of (word_idxs, char_idxs, tag_idxs).
    We pad along the seq_len dimension to produce:
      word_padded => (batch_size, max_seq_len)
      char_padded => (batch_size, max_seq_len, max_char_len)
      tag_padded  => (batch_size, max_seq_len)
      mask        => (batch_size, max_seq_len)  1=valid, 0=pad
    """
    # 1. Separate the lists
    word_list = [item[0] for item in batch]  # each shape (seq_len,)
    char_list = [item[1] for item in batch]  # each shape (seq_len, max_char_len)
    tag_list  = [item[2] for item in batch]  # each shape (seq_len,)

    # 2. Pad word_list => shape (batch_size, max_seq_len)
    word_padded = torch.nn.utils.rnn.pad_sequence(
        word_list, batch_first=True, padding_value=0  # assuming 0 is PAD index in word vocab
    )
    # 3. Pad tag_list => shape (batch_size, max_seq_len)
    tag_padded = torch.nn.utils.rnn.pad_sequence(
        tag_list, batch_first=True, padding_value=0  # assuming 0 is PAD index in tag vocab
    )

    # 4. Pad char_list => shape (batch_size, max_seq_len, max_char_len)
    max_seq_len = word_padded.size(1)
    padded_chars = []
    for chars2d in char_list:
        seq_len = chars2d.size(0)
        if seq_len < max_seq_len:
            diff = max_seq_len - seq_len
            # pad_2d => (left, right, top, bottom) for 2D
            pad_2d = (0, 0, 0, diff)  # only pad "down" the seq_len dimension
            chars2d = torch.nn.functional.pad(chars2d, pad_2d, mode='constant', value=0)
        padded_chars.append(chars2d)
    char_padded = torch.stack(padded_chars, dim=0)

    # 5. Build a mask where tag != 0 => 1, else 0
    #    (Adjust if your actual PAD index in the tag vocab is not 0!)
    mask = (tag_padded != 0).long()

    return word_padded, char_padded, tag_padded, mask


def main():
    ########################################################################
    # 1) INIT DISTRIBUTED
    ########################################################################
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    ########################################################################
    # 2) LOAD + SPLIT DATA
    ########################################################################
    train_data = load_data("A2-data/train")
    val_data   = load_data("A2-data/dev.answers")
    test_data  = load_data("A2-data/test_answers/test.answers")

    train_dataset, val_dataset, test_dataset = createTagSets(train_data, val_data, test_data)

    ########################################################################
    # 3) DISTRIBUTED SAMPLERS
    ########################################################################
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False, drop_last=False)
    test_sampler  = DistributedSampler(test_dataset,  shuffle=False, drop_last=False)

    ########################################################################
    # 4) DATALOADERS (now returning word_batch, char_batch, tag_batch, mask)
    ########################################################################
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        sampler=val_sampler,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        sampler=test_sampler,
        collate_fn=collate_fn
    )

    ########################################################################
    # 5) MODEL + DDP
    ########################################################################
    model = BiLSTM_CRF(
        word_vocab_size=len(train_dataset.token_vocab),
        char_vocab_size=len(train_dataset.char_vocab),
        tag_to_ix=train_dataset.tag_vocab,
        word_emb_dim=200,
        char_emb_dim=30,
        char_cnn_out_dim=50,
        hidden_dim=256
    ).to(device)

    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # -- SWITCH FROM SGD TO ADAM FOR FASTER CONVERGENCE
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    num_epochs = 10

    ########################################################################
    # 6) TRAINING LOOP
    ########################################################################
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0.0

        for word_batch, char_batch, tag_batch, mask in train_loader:
            word_batch = word_batch.to(device)
            char_batch = char_batch.to(device)
            tag_batch  = tag_batch.to(device)
            mask       = mask.to(device)

            optimizer.zero_grad()
            loss = ddp_model.module.neg_log_likelihood(word_batch, char_batch, tag_batch, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Let only rank 0 print
        if local_rank == 0:
            print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

        ####################################################################
        # Validation
        ####################################################################
        ddp_model.eval()
        val_sampler.set_epoch(epoch)
        val_loss = 0.0
        with torch.no_grad():
            for word_batch, char_batch, tag_batch, mask in val_loader:
                word_batch = word_batch.to(device)
                char_batch = char_batch.to(device)
                tag_batch  = tag_batch.to(device)
                mask       = mask.to(device)

                loss = ddp_model.module.neg_log_likelihood(word_batch, char_batch, tag_batch, mask)
                val_loss += loss.item()

        if local_rank == 0:
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

    ########################################################################
    # 7) TEST / INFERENCE
    ########################################################################
    ddp_model.eval()
    test_sampler.set_epoch(0)
    if local_rank == 0:
        print("Running test set evaluation...")

    tag_idx2label = test_dataset.get_tag_vocab_inverse()
    output_file = "test_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        model.eval()
        with torch.no_grad():
            for word_batch, char_batch, tag_batch, mask in test_loader:
                word_batch = word_batch.to(device)
                char_batch = char_batch.to(device)
                tag_batch  = tag_batch.to(device)
                mask       = mask.to(device)

                scores, pred_batch = ddp_model.module(word_batch, char_batch, mask)
                # pred_batch => (batch_size, seq_len)
                # Convert each predicted ID to a tag string and write it out
                batch_size, seq_len = pred_batch.shape

                for b in range(batch_size):
                    for s in range(seq_len):
                        # skip printing pad positions
                        if mask[b, s] == 0:
                            break
                        pred_tag_id = pred_batch[b, s].item()
                        tag_str = tag_idx2label[pred_tag_id]
                        f.write(tag_str + "\n")
                    f.write("\n")

    print(f"Predicted labels written to {output_file}")


def run_distributed():
    main()

if __name__ == "__main__":
    run_distributed()
