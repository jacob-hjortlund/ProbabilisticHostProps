from pathlib import Path
import paths as paths

NULL_VALUE = -9999.0
PECULIAR_VELOCITY_DISPERSION = 200.0  # km/s
SPEED_OF_LIGHT = 300000.0  # 299792.458 # km/s


def save_variable(
    variable: float,
    name: str,
    save_path: Path = paths.data,
    precision: int = 3,
    dtype: str = "float",
) -> None:
    """Saves a variable to a .txt file. TODO: Add Error Based Rounding.

    Args:
        variable (float): Variable to save.
        name (str): Name of variable.
        save_path (str, optional): Path to save variable. Defaults to SAVE_PATH.
        precision (int, optional): Precision of variable. Defaults to 3.
        dtype (str, optional): Data type of variable. Defaults to "float".
    """
    if type(save_path) == str:
        save_path = Path(save_path)
    with open(save_path / f"{name}.txt", "w") as f:

        if dtype == "float":
            string = f"{variable:.{precision}f}"
        elif dtype == "int":
            string = f"{int(variable)}"
        else:
            raise ValueError("dtype must be either float or int.")

        f.write(string)
