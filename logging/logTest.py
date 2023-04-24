import logging

logging.basicConfig(level=logging.INFO)

# 将根记录器级别设置为指定的级别。默认生成的 root logger 的 level 是 logging.WARNING，
# 低于该级别的就不输出了。级别排序：CRITICAL > ERROR > WARNING > INFO > DEBUG。（如果需要显示所有级别的内容，可将 level=logging.NOTSET）

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='test.log',  # log 保存目录
                    filemode='a')


logging.debug("debug")
logging.info('info')
logging.warning('warning')
logging.error('error')
logging.critical('critcal')

log = logging.getLogger()
log.info("Initialize done!!!")
log.critical('cccccccccccc')

