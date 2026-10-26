import desktop


def test_server_command_is_loopback_only():
    command = desktop.server_command(8509)
    assert "--server.address=127.0.0.1" in command
    assert "--server.port=8509" in command
    assert command[1:4] == ["-m", "streamlit", "run"]
