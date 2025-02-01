#import dailsathi_en
#import dial_hi
#import dailsathi_ka
#import dailsathi_tn
#import dailsathi_tel
#import dailsathi_mal


def main():
    print("Select a language pipeline:")
    print("1. English")
    print("2. Hindi")
    print("3. Kannada")


    choice = input("Enter your choice (1-6): ").strip()

    if choice == "1":
        import dailsathi_en
        dailsathi_en.main()
    elif choice == "2":
        import dial_hi
        dial_hi.main()
    elif choice == "3":
        import dailsathi_ka
        dailsathi_ka.main()
    else:
        print("Invalid choice. Please select a number between 1 and 6.")


if __name__ == "__main__":
    main()