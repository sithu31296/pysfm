from pysfm import *
from pysfm.retrieval import ImageRetrieval
from pysfm.utils.imageio import read_image, read_image_paths, read_images



def main(query_dir, database_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_image_paths = read_image_paths(query_dir)
    database_image_paths = read_image_paths(database_dir)

    retrieval = ImageRetrieval("netvlad", device, save_dir)
    retrieval(query_image_paths, database_image_paths)
    

    


if __name__ == '__main__':
    query = "./assets/tower/query"
    database = "./assets/tower/database"
    save_dir = Path("./assets/tower")
    save_dir.mkdir(parents=True, exist_ok=True)

    main(query, database, save_dir)