class ApiModelError(Exception):
    """modling error for the API
    defined after the example: https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
    """

    # default status code for API model error is 550

    status_code: int = 550

    def __init__(self, message: str, status_code: int | None = None, *args: object) -> None:
        super().__init__(message, *args)
        self.message: str = message
        if status_code is not None:
            self.status_code = status_code

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, str]:
        return {"message": self.message}


def translate_key(bracket_key: str) -> str:
    "converts key from <Variable [module]> to <Variable // module"
    if "[" not in bracket_key and "]" not in bracket_key:
        return bracket_key
    splitname = bracket_key.split("[")
    varname = (splitname[0]).strip()
    modname = (splitname[1].split("]")[0]).strip()

    return f"{varname} // {modname}"