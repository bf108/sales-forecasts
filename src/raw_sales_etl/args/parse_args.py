from argparse import ArgumentParser


def create_arg_parser() -> ArgumentParser:
    """
    Create the argument parser for the program
    :return: The argument parser
    """
    parser = ArgumentParser(
        prog="raw_sales_etl", description="Prepare raw sales data for forecasting"
    )
    return parser
