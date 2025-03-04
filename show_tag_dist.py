from collections import Counter

def show_tag_distribution(file_path):
    """
    Reads the file located at file_path and shows the distribution of tags.
    Adjust the parsing logic below to reflect how tags actually appear in your file.
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        arr = []
        for line in f:
            item = line.split('\t')
            if len(item)==2:
                arr.append(item[1])
            
        print(len(arr))
        x =Counter(arr)
        print(x)

    # Print the tag distribution
    # print("Tag Distribution (tag => count):")
    # for tag, count in tag_counter.most_common():
    #     print(f"{tag} => {count}")


if __name__ == "__main__":
    file_path = "A2-data/test_answers/test.answers"  # Adjust as needed
    show_tag_distribution(file_path)
