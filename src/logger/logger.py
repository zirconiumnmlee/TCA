import os
import logging

def setup_logger(query, log_folder: str):
    """
    query: log name
    log_folder: log output folder
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_filename = os.path.join(
        log_folder, f"{query.replace(' ', '_')}.log"
    )
    logger = logging.getLogger(query)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

if __name__ == "__main__":
    print("=== Logger Test Program ===")

    test_query = "Test Query"
    log_folder = "./test_logs"

    print(f"Creating logger, query: {test_query}")
    print(f"Log folder: {log_folder}")

    logger = setup_logger(test_query, log_folder)

    logger.info("This is an info log")
    logger.warning("This is a warning log")
    logger.error("This is an error log")

    logger.info("Starting to process user request")
    logger.info("Validating user permissions")
    logger.info("Executing database query")
    logger.warning("Query response time is long")
    logger.info("Returning query results")
    logger.info("Request processing completed")

    print(f"\nLogs saved to: {log_folder}")
    print("Test completed!")