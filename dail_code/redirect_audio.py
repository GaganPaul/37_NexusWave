#import dailsathi_en
#import dial_hi
#import dailsathi_ka
#import dailsathi_tn
#import dailsathi_tel
#import dailsathi_mal


def main(path :str,language :str,unique_filename :str):
    # print("Select a language pipeline:")
    # print("1. English")
    # print("2. Hindi")
    # print("3. Kannada")
    # print("4. Tamil")
    # print("5. Telugu")
    # print("6. Malayalam")
    # choice = input("Enter your choice (1-6): ").strip()

    print(f"Path={path}, Language={language}, unique_filename: {unique_filename}")

    if language == "1":
        import dailsathi_en
        return str(dailsathi_en.main(path=path, unique_filename=unique_filename))
    elif language == "2":
        import dial_hi
        return str(dial_hi.main(path=path, unique_filename=unique_filename))
    elif language == "3":
        import dailsathi_ka
        return str(dailsathi_ka.main(path=path, unique_filename=unique_filename))
    elif language == "4":
        import dailsathi_tn
        return str(dailsathi_tn.main(path=path, unique_filename=unique_filename))
    elif language == "5":
        import dailsathi_tel
        return str(dailsathi_tel.main(path=path, unique_filename=unique_filename))
    elif language == "6":
        import dailsathi_mal
        return str(dailsathi_mal.main(path=path, unique_filename=unique_filename))
    else:
        print(language)
        print("Invalid choice. Please select a number between 1 and 6.")


# if __name__ == "__main__":
#     main()