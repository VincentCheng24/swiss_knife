from splinter.browser import Browser
from apscheduler.schedulers.blocking import BlockingScheduler


class HackResigration(object):
    def __init__(self):
        self.url = 'https://www.muenchen.de/rathaus/terminvereinbarung_fs.html'

        self.driver_name = 'chrome'
        self.executable_path = '/Users/vincent/Desktop/chromedriver'

    def register(self):
        driver = Browser(driver_name=self.driver_name, executable_path=self.executable_path)

        driver.driver.set_window_size(700, 1000)

        driver.visit(self.url)

        # try:
        with driver.get_iframe('appointment') as iframe:

            iframe.execute_script("javascript:toggle('Umschreibung_SPACE_eines_SPACE_ausländischen_SPACE_Führerscheins');")
            iframe.select('CASETYPES[FS Umschreibung Ausländischer FS]', "1")
            iframe.find_by_xpath("//input[@value='Weiter']").first.click()

        print('well')
        # except:
        #     pass


hacker = HackResigration()

scheduler = BlockingScheduler()
scheduler.add_job(hacker.register(), 'interval', minutes=1)
scheduler.start()

