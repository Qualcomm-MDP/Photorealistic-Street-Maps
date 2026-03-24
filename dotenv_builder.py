from pathlib import Path

ENV_OPTIONS = {
    "1": ("development", ".env.development"),
    "2": ("production", ".env.production"),
    "3": ("none", ".env"),
}


def choose_environment() -> tuple[str, Path]:
    print("Select environment:")
    print("  1) development  →  .env.development")
    print("  2) production   →  .env.production")
    print("  3) none         →  .env")

    while True:
        choice = input("Choice (1/2/3): ").strip()
        if choice in ENV_OPTIONS:
            label, filename = ENV_OPTIONS[choice]
            print(f"Building {filename} for [{label}]")
            return label, Path(f"./{filename}")
        print("Invalid choice, enter 1, 2, or 3.")


def main():
    dotenv_example_path = Path("./.env.example")

    if not dotenv_example_path.is_file():
        raise FileNotFoundError(f"{dotenv_example_path.name} not found in directory!")

    label, dotenv_path = choose_environment()

    if dotenv_path.is_file():
        overwrite = (
            input(f"{dotenv_path.name} already exists. Overwrite? (y/N): ")
            .strip()
            .lower()
        )
        if overwrite != "y":
            print("Aborted.")
            return

    envs = {}
    print()
    with open(dotenv_example_path) as dotenv_example:
        for line in dotenv_example:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            user_value = input(f"Input value for {key}: ").strip()
            envs[key] = user_value

    with open(dotenv_path, "w") as dotenv_file:
        for key, value in envs.items():
            dotenv_file.write(f'{key}="{value}"\n')

    if label != "none":
        print(f"Remember to set APP_ENV={label} in your shell before running the app.")
    print(f"{dotenv_path.name} written with {len(envs)} variable(s).")


if __name__ == "__main__":
    main()
