import pickle
import pprint
import argparse

def read_pickle(file_path):
    """
    Read a pickle file and return its content.
    
    Args:
        file_path (str): Path to the pickle file
    
    Returns:
        The unpickled object
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return None

def display_pickle_content(file_path):
    """
    Read a pickle file and pretty print its content.
    
    Args:
        file_path (str): Path to the pickle file
    """
    data = read_pickle(file_path)
    if data is not None:
        print("Pickle file content:")
        pprint.pprint(data)

if __name__ == "__main__":
    filename = "hybrid_statistics_discrete_07-03_08:28:35.pickle"
    # parser = argparse.ArgumentParser(description='Read and display pickle file content')
    # parser.add_argument('file_path', type=str, help='Path to the pickle file', default=filename)
    # args = parser.parse_args()
    
    
    display_pickle_content(filename)