from faker import Faker
class FakerProvider:
    def __init__(self):
        self.faker = Faker("zh_TW")
        self.cache = {}

    def _generate(self, etype):
        match etype:
            case "NAME":   return self.faker.name()
            case "ID":     return self.faker.ssn(min_age=18, max_age=60)
            case "PHONE":  return self.faker.phone_number()
            case "EMAIL":  return self.faker.email()
            case _ :       return self.faker.word()

    def fake(self, etype:str, original:str):
        key = f"{etype}:{original}"
        if key not in self.cache:
            self.cache[key] = self._generate(etype)
        return self.cache[key]
