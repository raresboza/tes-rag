from backend.database_utils.setup import setup_vector_store_manager

def main():
    setup_vector_store_manager('data/wiki_links.csv')

if __name__ == "__main__":
    main()
