class Request():
    def __init__(self, role, message):
        self.role = role
        self.message = message

    def get_body(self):
        return {"role": self.role, "content": self.message}
