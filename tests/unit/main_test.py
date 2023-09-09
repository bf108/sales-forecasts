from pytest import CaptureFixture

from raw_sales_etl.main import main


def test_main(capfd: CaptureFixture[str]) -> None:
    """main() should print "Hello World!"."""
    # act
    main()

    # assert
    out, err = capfd.readouterr()
    assert out == "Hello World!\n"
    assert err == ""
