import torch
from torchvision import transforms

def mnist():
    train_data, train_labels = [],[]
    transform= transforms.Normalize((0,), (1,))

    for i in range(5):
        train_data.append(torch.load(f'./data/raw/train_images_{i}.pt'))
        train_labels.append(torch.load(f'./data/raw/train_target_{i}.pt'))

    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)

    train_data = transform(train_data)

    train_data = train_data.unsqueeze(1)

    torch.save(train_data, './data/processed/train_data.pt')
    torch.save(train_labels, './data/processed/train_labels.pt')

    print(f'Train data shape: {train_data.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print("Data successfully processed, normalized and saved")

if __name__ == '__main__':
    # Get the data and process it
    mnist()
    pass

