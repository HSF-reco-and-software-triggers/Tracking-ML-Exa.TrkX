class Bar:
    """
    A test class
    """
    def __init__(self):
        self.hello = "Hi there"

    def test_method(self, new_hello):
        """
        A test method
        """
        self.hello = new_hello
