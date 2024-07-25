from DAO.ContentMarketMongoDAO import ContentMarketMongoDAO
from DAO.ContentSpaceMongoDAO import ContentSpaceMongoDAO
from DAO.ContentDemandSupplyMongoDAO import ContentDemandSupplyMongoDAO

from typing import Dict


class DAOFactory:
    def get_content_market_dao(self, db_config: Dict[str, str]) \
            -> ContentMarketMongoDAO:
        """Return ContentMarketDAO by <db_config> specification.
        """
        if db_config["db_type"] == "Mongo":
            return ContentMarketMongoDAO(**db_config)
        else:
            raise ValueError

    def get_content_space_dao(self, db_config: Dict[str, str]) \
            -> ContentSpaceMongoDAO:
        """Return ContentSpaceMongoDAO by <db_config> specification.
        """
        if db_config["db_type"] == "Mongo":
            return ContentSpaceMongoDAO(**db_config)
        else:
            raise ValueError

    def get_supply_demand_dao(self, db_config: Dict[str, str]) \
            -> ContentDemandSupplyMongoDAO:
        """Return ContentDemandSupplyMongoDAO by <db_config> specification.
        """
        if db_config["db_type"] == "Mongo":
            return ContentDemandSupplyMongoDAO(**db_config)
        else:
            raise ValueError
